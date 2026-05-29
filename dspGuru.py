"""
BY DEEPSEEK
Adapter for HeartsAI to work with the existing pipeline (game.py, models.py).
Only implements evaluation mode (val=True), no gradient/log_prob required.
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add pipeline directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from hearts_game import Card, Suit, Rank, HeartsGame
from hearts_ai import HeartsAI, AIWeights, load_weights, GameMode

# Mapping between pipeline suit indices and hearts_game Suit enum
# pipeline suit: 0=Spades, 1=Hearts, 2=Clubs, 3=Diamonds
PIPELINE_TO_SUIT = {
    0: Suit.SPADES,
    1: Suit.HEARTS,
    2: Suit.CLUBS,
    3: Suit.DIAMONDS,
}
SUIT_TO_PIPELINE = {v: k for k, v in PIPELINE_TO_SUIT.items()}

# Rank mapping: pipeline rank 0..12 -> 2..A (value 2..14)
def pipeline_rank_to_rank(prank: int) -> Rank:
    """Convert pipeline rank (0=2, 12=A) to Rank enum."""
    value = prank + 2
    for r in Rank:
        if r.value == value:
            return r
    raise ValueError(f"Invalid pipeline rank: {prank}")

def rank_to_pipeline_rank(rank: Rank) -> int:
    """Convert Rank enum to pipeline rank (0=2, 12=A)."""
    return rank.value - 2


def pipeline_card_to_card(pcard: Tuple[int, int]) -> Card:
    """Convert (suit_idx, rank_idx) pipeline representation to Card."""
    suit_idx, rank_idx = pcard
    return Card(PIPELINE_TO_SUIT[suit_idx], pipeline_rank_to_rank(rank_idx))

def card_to_pipeline_card(card: Card) -> Tuple[int, int]:
    """Convert Card to pipeline representation (suit_idx, rank_idx)."""
    return (SUIT_TO_PIPELINE[card.suit], rank_to_pipeline_rank(card.rank))


def pipeline_hand_to_cards(hand: List[List[int]]) -> List[Card]:
    """Convert pipeline hand representation (list of 4 lists of ranks) to list of Cards."""
    cards = []
    for suit_idx, rank_list in enumerate(hand):
        for rank_idx in rank_list:
            cards.append(pipeline_card_to_card((suit_idx, rank_idx)))
    return cards

def cards_to_pipeline_hand(cards: List[Card]) -> List[List[int]]:
    """Convert list of Cards to pipeline hand representation."""
    hand = [[], [], [], []]
    for card in cards:
        suit_idx = SUIT_TO_PIPELINE[card.suit]
        rank_idx = rank_to_pipeline_rank(card.rank)
        hand[suit_idx].append(rank_idx)
    for suit_idx in range(4):
        hand[suit_idx].sort()
    return hand


class HeartsAIPlayer:
    """
    Wrapper to use HeartsAI in the existing pipeline's player interface.
    Implements: passcards, play, getcards, getpass, gettrick.
    """
    
    def __init__(self, player_index: int, 
                 num_players: int = 4,
                 weights: Optional[AIWeights] = None,
                 mode: GameMode = GameMode.PLAYER_4):
        self.player_index = player_index
        self.num_players = num_players
        self.mode = mode
        
        if weights is None:
            # Load default weights for the given mode
            all_weights = load_weights()
            weights = all_weights.get(mode, AIWeights())
        self.ai = HeartsAI(weights=weights)
        self.ai.reset_round(num_players)
        
        # Internal tracking for hearts_broken (since pipeline doesn't pass it directly)
        self.hearts_broken = False
        
    # ========== Required pipeline interface ==========
    
    def getcards(self, cards: List[List[int]]) -> None:
        """Called at start of round with the player's hand."""
        # Reset round state for HeartsAI
        self.ai.reset_round(self.num_players)
        self.hearts_broken = False
        
        # Store hand for possible later use (not strictly needed, but for consistency)
        self._current_hand = cards
        
    def getpass(self, cards: List[List[int]]) -> None:
        """Called after receiving passed cards (not needed for logic, but can be stored)."""
        # HeartsAI doesn't need this directly; the hand will be updated via later getcards
        # Just store for reference
        self._current_hand = cards
        
    def gettrick(self, trick_cards: List[Tuple[int, int]]) -> None:
        """
        Called after a trick is completed.
        trick_cards: list of (suit, rank) in playing order.
        """
        # Convert to Card objects
        cards = [pipeline_card_to_card(pc) for pc in trick_cards]
        # We don't know the winner directly from this call, but pipeline's Game
        # will also call record_trick_for_all separately if we want.
        # For simplicity, we can update tracker using the trick info.
        # However, we don't know winner index here. The adapter will also be called
        # via record_trick_for_all if we use that from pipeline.
        # We'll maintain our own trick recording via record_trick_from_pipeline.
        pass
    
    def passcards(self, cards: List[List[int]], direction_code: int, val: bool = False):
        """
        Select 3 cards to pass.
        cards: pipeline hand representation
        direction_code: 1 (left), 3 (right), 2 (across)
        val: if True, return only cards; if False, also return log_prob (ignored here)
        """
        # Convert direction code to string expected by HeartsAI
        if self.num_players == 4:
            dir_map = {1: "LEFT", 3: "RIGHT", 2: "ACROSS"}
        else:  # 3 players: only LEFT and RIGHT?
            # For 3-player, we need to handle properly. For now assume 4.
            dir_map = {1: "LEFT", 3: "RIGHT", 2: "ACROSS"}
        direction = dir_map.get(direction_code, "NONE")
        
        hand_cards = pipeline_hand_to_cards(cards)
        selected = self.ai.select_pass_cards(hand_cards, direction)
        
        # Convert back to pipeline representation
        result = [card_to_pipeline_card(c) for c in selected]
        
        return result
    
    def play(self, played: List[int], stat: List[int], now: List[Tuple[int, int]], 
             cards: List[List[int]], val: bool = False):
        """
        Select a card to play.
        played: 52-element list, 1 if card already played
        stat: 20-element list of statistics (voids, etc.)
        now: current trick cards so far, list of (suit, rank)
        cards: current player's hand (pipeline format)
        val: if True, return only card; if False, also return log_prob (ignored)
        """
        # 1. Determine hearts_broken status
        # hearts suit = 1, indices 13..25 in played array
        if not self.hearts_broken:
            hearts_played = any(played[13 + i] for i in range(13))
            if hearts_played:
                self.hearts_broken = True
        
        # 2. Convert hand to Card list
        hand_cards = pipeline_hand_to_cards(cards)
        
        # 3. Build Trick object from 'now' and known lead suit
        trick = self._build_trick(now)
        
        # 4. Get valid plays using pipeline's getavailable function
        from game import getavailable  # import here to avoid circular dependency
        available_by_suit = getavailable(self.hearts_broken, now, cards)
        # Convert to list of Card objects
        valid_plays = []
        for suit_idx, rank_list in enumerate(available_by_suit):
            for rank_idx in rank_list:
                valid_plays.append(pipeline_card_to_card((suit_idx, rank_idx)))
        
        # 5. Call HeartsAI's select_play
        selected = self.ai.select_play(
            hand=hand_cards,
            valid_plays=valid_plays,
            trick=trick,
            player_index=self.player_index,
            num_players=self.num_players,
            hearts_broken=self.hearts_broken
        )
        
        # 6. Convert back to pipeline representation
        return card_to_pipeline_card(selected)
    
    # ========== Helper methods ==========
    
    def _build_trick(self, now: List[Tuple[int, int]]):# -> 'Trick':
        """Build a Trick object from pipeline's 'now' list."""
        from hearts_game import Trick
        if not now:
            # Leading, trick is empty
            return Trick(lead_player_index=self.player_index, num_players=self.num_players)
        
        # Convert cards to (player_index, Card) - we don't know player indices from 'now'
        # For HeartsAI, the Trick object mostly needs lead_suit and the cards list.
        # Player indices are not strictly required for decision logic (except maybe position),
        # but we can assign sequential indices starting from the lead player.
        lead_suit = PIPELINE_TO_SUIT[now[0][0]]
        trick = Trick(lead_player_index=0, num_players=self.num_players)
        # We don't have real player indices, so we assign dummy indices.
        # The AI uses trick.cards only to check if a card is in the trick (which is not needed)
        # and to get lead_suit, highest card, etc. It doesn't rely on absolute indices.
        for i, (suit_idx, rank_idx) in enumerate(now):
            card = pipeline_card_to_card((suit_idx, rank_idx))
            trick.add_card(i, card)
        return trick
    
    def record_trick_info(self, trick_info: List[Tuple[int, Tuple[int, int]]], winner_index: int):
        """
        Optional: if the pipeline provides full trick info (player indices and cards),
        this can be used to update the AI's tracker.
        """
        # Convert trick_info from pipeline format to list of (player_index, Card)
        converted = [(pid, pipeline_card_to_card(pcard)) for pid, pcard in trick_info]
        trick_cards = [card for _, card in converted]
        self.ai.record_trick(trick_cards, winner_index, converted)
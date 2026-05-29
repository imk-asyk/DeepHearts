# Created by dspGuru

"""
Hearts card game implementation.

Standard rules:
- 3 or 4 players
- 4 players: 52 cards, 13 cards each
- 3 players: 51 cards (2♦ removed), 17 cards each
- Pass 3 cards (rotation varies by player count)
- 2 of clubs leads first trick (or 3 of clubs in 3-player if 2♦ removed)
- Must follow suit if possible
- Hearts cannot be led until broken
- Hearts = 1 point each, Queen of Spades = 13 points
- Shooting the moon: take all hearts + QoS = 0 points, others get 26
- Game ends when a player reaches 100 points
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import random


class Suit(Enum):
    CLUBS = 0
    DIAMONDS = 1
    SPADES = 2
    HEARTS = 3

    def __str__(self) -> str:
        symbols = {
            Suit.CLUBS: "♣",
            Suit.DIAMONDS: "♦",
            Suit.SPADES: "♠",
            Suit.HEARTS: "♥",
        }
        return symbols[self]


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def __str__(self) -> str:
        names = {
            Rank.TWO: "2",
            Rank.THREE: "3",
            Rank.FOUR: "4",
            Rank.FIVE: "5",
            Rank.SIX: "6",
            Rank.SEVEN: "7",
            Rank.EIGHT: "8",
            Rank.NINE: "9",
            Rank.TEN: "10",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }
        return names[self]


class PassDirection(Enum):
    LEFT = auto()
    RIGHT = auto()
    ACROSS = auto()
    NONE = auto()


class ThreePlayerMode(Enum):
    """How to handle the extra card in 3-player games."""

    REMOVE_CARD = auto()  # Remove 2♦ from deck (default)
    KITTY = auto()  # Set aside one card; first player to take points gets it


class GameMode(Enum):
    """Specific game modes for AI heuristic tuning."""

    PLAYER_4 = auto()
    PLAYER_3_REMOVE = auto()
    PLAYER_3_KITTY = auto()

    @staticmethod
    def from_settings(
        num_players: int,
        three_player_mode: ThreePlayerMode = ThreePlayerMode.REMOVE_CARD,
    ) -> "GameMode":
        """Determine game mode from settings."""
        if num_players == 4:
            return GameMode.PLAYER_4
        elif num_players == 3:
            if three_player_mode == ThreePlayerMode.REMOVE_CARD:
                return GameMode.PLAYER_3_REMOVE
            else:
                return GameMode.PLAYER_3_KITTY
        return GameMode.PLAYER_4  # Default


class GamePhase(Enum):
    DEALING = auto()
    PASSING = auto()
    PLAYING = auto()
    ROUND_END = auto()
    GAME_OVER = auto()


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def points(self) -> int:
        """Return the point value of this card."""
        if self.suit == Suit.HEARTS:
            return 1
        if self.suit == Suit.SPADES and self.rank == Rank.QUEEN:
            return 13
        return 0

    @property
    def is_queen_of_spades(self) -> bool:
        return self.suit == Suit.SPADES and self.rank == Rank.QUEEN

    @property
    def is_two_of_clubs(self) -> bool:
        return self.suit == Suit.CLUBS and self.rank == Rank.TWO


@dataclass
class Player:
    name: str
    hand: list[Card] = field(default_factory=list)
    taken_cards: list[Card] = field(default_factory=list)
    round_score: int = 0
    total_score: int = 0
    cards_to_pass: list[Card] = field(default_factory=list)

    def reset_for_round(self) -> None:
        """Reset player state for a new round."""
        self.hand.clear()
        self.taken_cards.clear()
        self.round_score = 0
        self.cards_to_pass.clear()

    def has_card(self, card: Card) -> bool:
        return card in self.hand

    def has_suit(self, suit: Suit) -> bool:
        return any(c.suit == suit for c in self.hand)

    def has_only_hearts(self) -> bool:
        return all(c.suit == Suit.HEARTS for c in self.hand)

    def get_cards_of_suit(self, suit: Suit) -> list[Card]:
        return [c for c in self.hand if c.suit == suit]

    def play_card(self, card: Card) -> Card:
        """Remove and return a card from the player's hand."""
        if card not in self.hand:
            raise ValueError(f"{self.name} does not have {card}")
        self.hand.remove(card)
        return card

    def receive_cards(self, cards: list[Card]) -> None:
        """Add cards to the player's hand."""
        self.hand.extend(cards)

    def take_trick(self, cards: list[Card]) -> None:
        """Add cards from a won trick to taken_cards."""
        self.taken_cards.extend(cards)

    def calculate_round_score(self) -> int:
        """Calculate points from taken cards."""
        self.round_score = sum(c.points for c in self.taken_cards)
        return self.round_score

    def sort_hand(self) -> None:
        """Sort hand by suit then rank."""
        self.hand.sort(key=lambda c: (c.suit.value, c.rank.value))


@dataclass
class Trick:
    lead_player_index: int
    num_players: int = 4
    cards: list[tuple[int, Card]] = field(default_factory=list)

    @property
    def lead_suit(self) -> Optional[Suit]:
        if not self.cards:
            return None
        return self.cards[0][1].suit

    @property
    def is_complete(self) -> bool:
        return len(self.cards) == self.num_players

    def add_card(self, player_index: int, card: Card) -> None:
        self.cards.append((player_index, card))

    def get_winner(self) -> int:
        """Return the index of the player who won the trick."""
        if not self.cards:
            raise ValueError("No cards in trick")

        lead_suit = self.lead_suit
        winning_player = self.cards[0][0]
        winning_rank = self.cards[0][1].rank

        for player_index, card in self.cards[1:]:
            if card.suit == lead_suit and card.rank.value > winning_rank.value:
                winning_player = player_index
                winning_rank = card.rank

        return winning_player

    def get_cards(self) -> list[Card]:
        """Return just the cards without player indices."""
        return [card for _, card in self.cards]

    def contains_points(self) -> bool:
        """Check if any card in the trick has points."""
        return any(card.points > 0 for _, card in self.cards)


class HeartsGame:
    """
    Implements the standard rules of the card game Hearts.

    Usage:
        game = HeartsGame(["Alice", "Bob", "Carol", "Dave"])  # 4 players
        game = HeartsGame(["Alice", "Bob", "Carol"])  # 3 players (removes 2♦)
        game = HeartsGame(
            ["Alice", "Bob", "Carol"],
            three_player_mode=ThreePlayerMode.KITTY  # Set aside 1 card
        )
        game.start_round()

        # During passing phase
        game.set_pass_cards(player_index, [card1, card2, card3])
        game.execute_pass()

        # During playing phase
        game.play_card(player_index, card)

        # Check game state
        game.get_valid_plays(player_index)
        game.is_game_over()

    Three-player modes:
        REMOVE_CARD (default): Remove 2♦ from deck, each player gets 17 cards
        KITTY: Set aside 1 random card; first player to take points gets it
    """

    WINNING_SCORE = 100
    CARDS_TO_PASS = 3

    # Pass directions vary by player count
    PASS_DIRECTIONS_4P = [
        PassDirection.LEFT,
        PassDirection.RIGHT,
        PassDirection.ACROSS,
        PassDirection.NONE,
    ]
    PASS_DIRECTIONS_3P = [
        PassDirection.LEFT,
        PassDirection.RIGHT,
        PassDirection.NONE,
    ]

    def __init__(
        self,
        player_names: list[str],
        three_player_mode: ThreePlayerMode = ThreePlayerMode.REMOVE_CARD,
    ):
        if len(player_names) not in (3, 4):
            raise ValueError("Hearts requires 3 or 4 players")

        self.names = player_names
        self.num_players = len(self.names)
        self.three_player_mode = three_player_mode
        self.game_mode = GameMode.from_settings(len(self.names), three_player_mode)
        self.players = [Player(name) for name in self.names]
        self.phase = GamePhase.DEALING
        self.current_trick: Optional[Trick] = None
        self.tricks_played = 0
        self.current_player_index = 0
        self.hearts_broken = False
        self.round_number = 0
        self.pass_direction = PassDirection.LEFT
        self.removed_card: Optional[Card] = None  # Track removed card for 3-player
        self.kitty_card: Optional[Card] = None  # Track set-aside card in KITTY mode
        self.kitty_claimed = False  # Whether kitty has been claimed this round

        # Set pass directions based on player count
        self.pass_directions = (
            self.PASS_DIRECTIONS_4P
            if self.num_players == 4
            else self.PASS_DIRECTIONS_3P
        )

    def _create_deck(self) -> list[Card]:
        """Create a deck appropriate for the number of players and mode.

        For 4 players: standard 52-card deck (13 cards each)
        For 3 players with REMOVE_CARD: 51 cards with 2♦ removed (17 cards each)
        For 3 players with KITTY: 52 cards, one set aside after shuffle (17 cards each)
        """
        deck = [Card(suit, rank) for suit in Suit for rank in Rank]

        if (
            self.num_players == 3
            and self.three_player_mode == ThreePlayerMode.REMOVE_CARD
        ):
            # Remove 2 of diamonds for 3-player game
            self.removed_card = Card(Suit.DIAMONDS, Rank.TWO)
            deck.remove(self.removed_card)

        return deck

    def start_round(self) -> None:
        """Start a new round: shuffle, deal, and set up passing phase."""
        # Reset state
        for player in self.players:
            player.reset_for_round()

        self.hearts_broken = False
        self.tricks_played = 0
        self.current_trick = None
        self.round_number += 1
        self.kitty_card = None
        self.kitty_claimed = False

        # Determine pass direction
        direction_index = (self.round_number - 1) % len(self.pass_directions)
        self.pass_direction = self.pass_directions[direction_index]

        # Shuffle and deal
        deck = self._create_deck()
        random.shuffle(deck)

        # In KITTY mode, set aside one card before dealing
        if self.num_players == 3 and self.three_player_mode == ThreePlayerMode.KITTY:
            # Ensure 2 of clubs is not the kitty card (it must be played to start)
            two_of_clubs = Card(Suit.CLUBS, Rank.TWO)
            while deck[-1] == two_of_clubs:
                random.shuffle(deck)
            self.kitty_card = deck.pop()

        cards_per_player = len(deck) // self.num_players
        for i, player in enumerate(self.players):
            start = i * cards_per_player
            end = start + cards_per_player
            player.receive_cards(deck[start:end])
            player.sort_hand()

        # Set phase
        if self.pass_direction == PassDirection.NONE:
            self.phase = GamePhase.PLAYING
            self._start_first_trick()
        else:
            self.phase = GamePhase.PASSING

    def set_pass_cards(self, player_index: int, cards: list[Card]) -> None:
        """Set the cards a player wants to pass."""
        if self.phase != GamePhase.PASSING:
            raise ValueError("Not in passing phase")
        if len(cards) != self.CARDS_TO_PASS:
            raise ValueError(f"Must pass exactly {self.CARDS_TO_PASS} cards")

        player = self.players[player_index]
        for card in cards:
            if card not in player.hand:
                raise ValueError(f"Player does not have {card}")

        player.cards_to_pass = list(cards)

    def all_players_ready_to_pass(self) -> bool:
        """Check if all players have selected cards to pass."""
        return all(len(p.cards_to_pass) == self.CARDS_TO_PASS for p in self.players)

    def execute_pass(self) -> None:
        """Execute the card passing between players."""
        if self.phase != GamePhase.PASSING:
            raise ValueError("Not in passing phase")
        if not self.all_players_ready_to_pass():
            raise ValueError("Not all players have selected cards to pass")

        # Calculate target indices based on direction and player count
        # For 4 players: left=1, right=-1(3), across=2
        # For 3 players: left=1, right=-1(2), no across
        if self.num_players == 4:
            offsets = {
                PassDirection.LEFT: 1,
                PassDirection.RIGHT: 3,  # -1 mod 4
                PassDirection.ACROSS: 2,
                PassDirection.NONE: 0,
            }
        else:  # 3 players
            offsets = {
                PassDirection.LEFT: 1,
                PassDirection.RIGHT: 2,  # -1 mod 3
                PassDirection.NONE: 0,
            }
        offset = offsets[self.pass_direction]

        # Collect cards to pass
        passing_cards = []
        for i, player in enumerate(self.players):
            for card in player.cards_to_pass:
                player.hand.remove(card)
            passing_cards.append(player.cards_to_pass)
            player.cards_to_pass = []

        # Distribute passed cards
        for i, player in enumerate(self.players):
            source_index = (i - offset) % self.num_players
            player.receive_cards(passing_cards[source_index])
            player.sort_hand()

        self.phase = GamePhase.PLAYING
        self._start_first_trick()

    def _start_first_trick(self) -> None:
        """Find the player with 2 of clubs and start the first trick."""
        two_of_clubs = Card(Suit.CLUBS, Rank.TWO)
        for i, player in enumerate(self.players):
            if player.has_card(two_of_clubs):
                self.current_player_index = i
                self.current_trick = Trick(
                    lead_player_index=i, num_players=self.num_players
                )
                return

        raise RuntimeError("No player has the 2 of clubs")

    def get_valid_plays(self, player_index: int) -> list[Card]:
        """Get the list of valid cards a player can play."""
        if self.phase != GamePhase.PLAYING:
            return []
        if player_index != self.current_player_index:
            return []

        player = self.players[player_index]
        hand = player.hand

        if not hand:
            return []

        # First card of the game must be 2 of clubs
        if self.tricks_played == 0 and not self.current_trick.cards:
            two_of_clubs = Card(Suit.CLUBS, Rank.TWO)
            return [two_of_clubs] if player.has_card(two_of_clubs) else []

        # Must follow suit if possible
        if self.current_trick.cards:
            lead_suit = self.current_trick.lead_suit
            cards_of_suit = player.get_cards_of_suit(lead_suit)
            if cards_of_suit:
                return cards_of_suit

            # Can't follow suit - can play anything, but...
            # On first trick, can't play hearts or queen of spades
            if self.tricks_played == 0:
                non_point_cards = [c for c in hand if c.points == 0]
                if non_point_cards:
                    return non_point_cards
            # Can play any card
            return list(hand)

        # Leading a trick
        if not self.hearts_broken:
            # Can't lead hearts unless hearts broken or only have hearts
            non_hearts = [c for c in hand if c.suit != Suit.HEARTS]
            if non_hearts:
                return non_hearts
            # Only have hearts - can lead them (breaks hearts)

        return list(hand)

    def play_card(self, player_index: int, card: Card) -> dict:
        """
        Play a card for the given player.

        Returns a dict with game state info:
        {
            'valid': bool,
            'trick_complete': bool,
            'trick_winner': Optional[int],
            'trick_cards': list[Card],
            'round_complete': bool,
            'game_over': bool,
            'error': Optional[str],
            'kitty_claimed': bool,  # True if kitty was claimed this trick
            'kitty_card': Optional[Card],  # The kitty card if claimed
            'trick_info': list[tuple[int, Card]]  # Pairings of (player_index, card)
        }
        """
        result = {
            "valid": False,
            "trick_complete": False,
            "trick_winner": None,
            "trick_cards": [],
            "round_complete": False,
            "game_over": False,
            "error": None,
            "kitty_claimed": False,
            "kitty_card": None,
            "trick_info": [],
        }

        if self.phase != GamePhase.PLAYING:
            result["error"] = "Not in playing phase"
            return result

        if player_index != self.current_player_index:
            result["error"] = f"Not player {player_index}'s turn"
            return result

        valid_plays = self.get_valid_plays(player_index)
        if card not in valid_plays:
            result["error"] = f"{card} is not a valid play"
            return result

        # Play the card
        player = self.players[player_index]
        player.play_card(card)
        self.current_trick.add_card(player_index, card)

        # Check if hearts broken
        if card.suit == Suit.HEARTS:
            self.hearts_broken = True

        result["valid"] = True

        # Check if trick is complete
        if self.current_trick.is_complete:
            result["trick_complete"] = True
            winner_index = self.current_trick.get_winner()
            result["trick_winner"] = winner_index
            result["trick_cards"] = self.current_trick.get_cards()
            result["trick_info"] = list(self.current_trick.cards)

            # Winner takes the trick
            winner = self.players[winner_index]
            trick_cards = self.current_trick.get_cards()
            winner.take_trick(trick_cards)

            # Recalculate round scores for all players to update real-time display
            for p in self.players:
                p.calculate_round_score()

            # In KITTY mode, first player to take points also gets the kitty
            if (
                self.kitty_card is not None
                and not self.kitty_claimed
                and self.current_trick.contains_points()
            ):
                winner.take_trick([self.kitty_card])
                result["kitty_claimed"] = True
                result["kitty_card"] = self.kitty_card
                self.kitty_claimed = True

            self.tricks_played += 1

            # Check if round is complete
            if self.tricks_played == self.tricks_per_round:
                result["round_complete"] = True
                self._end_round()

                if self.is_game_over():
                    result["game_over"] = True
                    self.phase = GamePhase.GAME_OVER
                else:
                    self.phase = GamePhase.ROUND_END
            else:
                # Start new trick
                self.current_trick = Trick(
                    lead_player_index=winner_index, num_players=self.num_players
                )
                self.current_player_index = winner_index
        else:
            # Move to next player
            self.current_player_index = (
                self.current_player_index + 1
            ) % self.num_players

        return result

    def _end_round(self) -> None:
        """Calculate scores at end of round, handling shooting the moon."""
        # Calculate round scores
        for player in self.players:
            player.calculate_round_score()

        # Check for shooting the moon
        moon_shooter = None
        for player in self.players:
            if player.round_score == 26:
                moon_shooter = player
                break

        if moon_shooter:
            # Shooter gets 0, everyone else gets 26
            for player in self.players:
                if player == moon_shooter:
                    player.round_score = 0
                else:
                    player.round_score = 26

        # Add to total scores
        for player in self.players:
            player.total_score += player.round_score

    def is_game_over(self) -> bool:
        """Check if any player has reached the winning score."""
        return any(p.total_score >= self.WINNING_SCORE for p in self.players)

    def get_winner(self) -> Optional[Player]:
        """Get the winner (lowest score when game ends)."""
        if not self.is_game_over():
            return None
        return min(self.players, key=lambda p: p.total_score)

    def get_scores(self) -> dict[str, int]:
        """Get current total scores for all players."""
        return {p.name: p.total_score for p in self.players}

    def get_round_scores(self) -> dict[str, int]:
        """Get current round scores for all players."""
        return {p.name: p.round_score for p in self.players}

    def get_current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.current_player_index]

    @property
    def tricks_per_round(self) -> int:
        """Number of tricks per round based on player count."""
        return 13 if self.num_players == 4 else 17

    def get_game_state(self) -> dict:
        """Get a summary of the current game state."""
        state = {
            "phase": self.phase.name,
            "num_players": self.num_players,
            "three_player_mode": (
                self.three_player_mode.name if self.num_players == 3 else None
            ),
            "round_number": self.round_number,
            "pass_direction": self.pass_direction.name,
            "current_player": self.players[self.current_player_index].name,
            "current_player_index": self.current_player_index,
            "tricks_played": self.tricks_played,
            "tricks_per_round": self.tricks_per_round,
            "hearts_broken": self.hearts_broken,
            "scores": self.get_scores(),
            "current_trick": (
                [str(c) for _, c in self.current_trick.cards]
                if self.current_trick
                else []
            ),
        }

        # Add kitty info for 3-player KITTY mode
        if self.num_players == 3 and self.three_player_mode == ThreePlayerMode.KITTY:
            state["kitty_claimed"] = self.kitty_claimed
            state["kitty_card"] = str(self.kitty_card) if self.kitty_card else None

        return state

# Created by dspGuru

"""
AI player for Hearts card game.

Implements strategic heuristics for passing and playing.
Each heuristic has configurable weights that can be adjusted.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, fields
from hearts_game import Card, Suit, Rank, Player, HeartsGame, Trick, GameMode


@dataclass
class AIWeights:
    """
    Configurable weights for AI heuristics.
    Higher values = higher priority for that behavior.

    PASSING WEIGHTS (used to prioritize which cards to pass):
    """

    # Pass the Queen of Spades when unprotected (very high - it's 13 points!)
    pass_queen_of_spades: float = 100.0

    # Minimum low spades needed to consider Q♠ "protected"
    queen_protection_threshold: int = 3

    # Pass cards to void a suit (enables future dumping)
    pass_to_void_suit: float = 80.0

    # Bonus for voiding minor suits (clubs/diamonds) vs major suits
    void_minor_suit_bonus: float = 5.0

    # Pass high hearts (they're worth points)
    pass_high_hearts: float = 70.0

    # Minimum rank to consider a heart "high" (10 = Ten)
    high_heart_threshold: int = 10

    # Pass high cards in other suits (Aces, Kings can win unwanted tricks)
    pass_high_cards: float = 50.0

    # Minimum rank to consider passing (12 = Queen)
    high_card_threshold: int = 12

    # Base priority for any card (rank value is added)
    pass_base_priority: float = 0.0

    """
    PASS DIRECTION WEIGHTS (different strategies based on direction):
    """
    # Penalty for passing high cards to player on left (they play after you)
    pass_left_high_card_penalty: float = 10.0

    # Bonus for voiding suits when passing right (they play before you)
    pass_right_void_bonus: float = 15.0

    # Bonus for passing dangerous cards across (furthest opponent)
    pass_across_danger_bonus: float = 5.0

    # Penalty for leading suits where you passed high cards
    avoid_passed_suit_penalty: float = 20.0

    # Pass the 2 of clubs to control the lead (or keep it if A is held)
    pass_two_of_clubs_control: float = 10.0

    # Bonus for retaining Ace of Clubs if we have 2 of Clubs
    retain_ace_of_clubs_bonus: float = 15.0

    # Bonus for "decoy" passes (mixing high and low cards)
    pass_decoy_bonus: float = 5.0

    # Penalty for passing the last card of a suit (signals void)
    pass_last_card_penalty: float = 15.0

    """
    PLAYING WEIGHTS (used during card play decisions):
    """
    # When leading, prefer these suits (higher = more preferred)
    lead_clubs_priority: float = 40.0
    lead_diamonds_priority: float = 30.0
    lead_spades_priority: float = 20.0  # Only after Q♠ gone
    lead_hearts_priority: float = 10.0  # Only after hearts broken

    # Prefer leading low cards (this multiplier reduces priority for high ranks)
    lead_low_card_preference: float = 2.0

    # When discarding, prioritize dumping these (higher = dump first)
    discard_queen_of_spades: float = 100.0
    discard_hearts: float = 80.0
    discard_dangerous_spades: float = 60.0  # A, K of spades when Q♠ not gone
    discard_high_cards: float = 40.0

    # Multiplier for rank when discarding (higher rank = higher discard priority)
    discard_rank_multiplier: float = 1.0

    """
    OPPONENT VOID TRACKING:
    """
    # Penalty for leading a suit where a dangerous opponent is void
    lead_avoid_void_opponent: float = 30.0

    # Bonus for leading suits where Q♠ holder is void
    lead_void_queen_holder: float = 25.0

    """
    SPADE FLUSH STRATEGY (smoke out the Queen):
    """
    # Priority for leading low spades to flush Q♠
    flush_queen_priority: float = 35.0

    # Maximum spade rank to use for flushing (8 = Eight)
    flush_queen_max_rank: int = 8

    # Minimum spades in hand to attempt flush
    flush_queen_min_spades: int = 2

    """
    A/K SPADE DANGER:
    """
    # Danger multiplier for A/K spades when not holding Q♠
    spade_ak_danger_multiplier: float = 1.5

    # Priority to pass A/K spades when dangerous
    pass_dangerous_ak_spades: float = 75.0

    # Weight for finesse opportunities (playing mid-rank to draw high)
    finesse_opportunity_weight: float = 12.0

    # Priority boost for actively leading Spades when Q is protected
    bleed_spades_priority: float = 30.0

    # Weight for dumping low hearts once moon control is established
    moon_dump_low_points_weight: float = 10.0

    # Risk weight for opponents holding high cards
    opponent_high_card_risk_weight: float = 15.0

    """
    SHOOT THE MOON OFFENSE:
    """
    # Minimum hand strength score to attempt moon (0-100)
    moon_attempt_threshold: float = 70.0

    # Points already collected to continue moon attempt
    moon_continue_threshold: int = 8

    # Priority for collecting points when shooting moon
    moon_collect_priority: float = 60.0

    # Weight for high hearts in moon hand evaluation
    moon_high_hearts_weight: float = 8.0

    # Weight for having Q♠ in moon hand evaluation
    moon_queen_spades_weight: float = 15.0

    # Weight for A♠/K♠ in moon hand evaluation
    moon_high_spades_weight: float = 10.0

    """
    EXIT CARD MANAGEMENT:
    """
    # Bonus for preserving low cards as exit routes
    preserve_exit_card_bonus: float = 15.0

    # Maximum rank considered an "exit card" (6 = Six)
    exit_card_threshold: int = 6

    # Minimum exit cards to try to preserve
    min_exit_cards: int = 2

    """
    SUIT CONTROL AND POSITION PLAY:
    """
    # Bonus for leading from a controlled suit (you have top cards)
    suit_control_lead_bonus: float = 20.0

    # Number of top cards needed for "control"
    control_card_count: int = 2

    # Preference to play low in second position
    second_hand_low_preference: float = 1.3

    # Preference to play high in third position to win
    third_hand_high_preference: float = 1.2

    """
    HEARTS DISTRIBUTION TRACKING:
    """
    # Preference for spreading hearts among players
    hearts_spread_preference: float = 10.0

    # Hearts in one player that triggers moon concern
    hearts_concentration_threshold: int = 6

    """
    LATE ROUND CAUTION:
    """
    # Multiplier for risk avoidance in late tricks
    late_round_caution_multiplier: float = 1.5

    # Tricks remaining to trigger late-round caution
    late_round_threshold: int = 4

    """
    ENDGAME PERFECT PLAY:
    """
    # Cards remaining to trigger endgame calculation
    endgame_threshold: int = 4

    # Enable endgame lookahead
    endgame_enabled: bool = True

    """
    MOON DEFENSE WEIGHTS:
    """
    # Points threshold to suspect someone is shooting the moon
    moon_threat_threshold: int = 10

    # Priority boost for blocking moon shooter
    moon_block_priority: float = 50.0

    """
    FOLLOWING SUIT WEIGHTS:
    """
    # When we can duck, prefer high duck (saves low cards for later)
    high_duck_preference: float = 1.0

    # When we must win, prefer low win (minimizes exposure)
    low_win_preference: float = 1.0

    # When last to play with no points, prefer taking trick (control)
    take_safe_trick_preference: float = 1.0


def load_weights(filename: str = "weights.json") -> dict[GameMode, AIWeights]:
    """Load weights from a JSON file."""
    weights = {
        GameMode.PLAYER_4: AIWeights(),
        GameMode.PLAYER_3_REMOVE: AIWeights(),
        GameMode.PLAYER_3_KITTY: AIWeights(),
    }

    path = Path(filename)
    if not path.exists():
        return weights

    try:
        with open(path, "r") as f:
            data = json.load(f)

        for mode_name, mode_weights in data.items():
            try:
                mode = GameMode[mode_name]
                # Filter weights to only include valid fields in AIWeights
                valid_fields = {f.name for f in fields(AIWeights)}
                filtered_weights = {
                    k: v for k, v in mode_weights.items() if k in valid_fields
                }
                weights[mode] = AIWeights(**filtered_weights)
            except (KeyError, ValueError):
                continue
    except Exception as e:
        print(f"Warning: Failed to load weights from {filename}: {e}")

    return weights


# Default weights instance (now mode-specific, loaded from JSON if available)
DEFAULT_WEIGHTS = load_weights()

# Global current weights (for all players in the current game)
_ai_weights = DEFAULT_WEIGHTS[GameMode.PLAYER_4]
_ai_mode = GameMode.PLAYER_4


@dataclass
class CardTracker:
    """Tracks which cards have been played."""

    played_cards: set[Card] = field(default_factory=set)
    void_players: dict[int, set[Suit]] = field(default_factory=dict)
    passed_cards: dict[int, list[Card]] = field(
        default_factory=dict
    )  # player_index -> cards passed TO them
    history: list[list[tuple[int, Card]]] = field(
        default_factory=list
    )  # list of tricks

    def reset(self):
        """Reset for a new round."""
        self.played_cards.clear()
        self.void_players.clear()
        self.passed_cards.clear()
        self.history.clear()

    def record_passed_cards(self, destination_player: int, cards: list[Card]):
        """Record cards passed to a specific player."""
        self.passed_cards[destination_player] = list(cards)

    def record_cards(self, cards: list[Card]):
        """Record cards that have been played."""
        self.played_cards.update(cards)

    def is_played(self, card: Card) -> bool:
        """Check if a card has been played."""
        return card in self.played_cards

    def remaining_in_suit(
        self, suit: Suit, excluding_hand: list[Card] = None
    ) -> list[Card]:
        """Get cards remaining in a suit (not yet played)."""
        excluding = set(excluding_hand) if excluding_hand else set()
        return [
            Card(suit, rank)
            for rank in Rank
            if Card(suit, rank) not in self.played_cards
            and Card(suit, rank) not in excluding
        ]

    def queen_of_spades_played(self) -> bool:
        """Check if Queen of Spades has been played."""
        return Card(Suit.SPADES, Rank.QUEEN) in self.played_cards

    def hearts_played_count(self) -> int:
        """Count how many hearts have been played."""
        return sum(1 for c in self.played_cards if c.suit == Suit.HEARTS)

    def is_player_void(self, player_index: int, suit: Suit) -> bool:
        """Check if a player is void in a suit."""
        return suit in self.void_players.get(player_index, set())

    def mark_player_void(self, player_index: int, suit: Suit):
        """Mark a player as void in a suit."""
        if player_index not in self.void_players:
            self.void_players[player_index] = set()
        self.void_players[player_index].add(suit)

    def get_high_card_probabilities(
        self, suit: Suit, num_players: int
    ) -> dict[int, float]:
        """
        Estimate probability of each player holding high cards in a suit.
        Returns player_index -> probability.
        """
        probs = {i: 0.0 for i in range(num_players)}
        remaining = self.remaining_in_suit(suit)
        if not remaining:
            return probs

        high_ranks = [Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK]
        high_remaining = [c for c in remaining if c.rank in high_ranks]

        if not high_remaining:
            return probs

        # Simple heuristic: players who haven't shown void have equal chance
        active_players = [
            i for i in range(num_players) if not self.is_player_void(i, suit)
        ]
        if not active_players:
            return probs

        base_prob = 1.0 / len(active_players)
        for i in active_players:
            probs[i] = base_prob

        # Adjust based on passed cards if we know them
        for player_idx, cards in self.passed_cards.items():
            for card in cards:
                if (
                    card.suit == suit
                    and card.rank in high_ranks
                    and not self.is_played(card)
                ):
                    probs[player_idx] += 0.5  # Boost if we passed it to them
                    # Normalize later if needed

        return probs


class HeartsAI:
    """
    Strategic AI for Hearts.

    Heuristics:
    - Passing: void suits, dump dangerous cards (Q♠, high hearts, aces/kings)
    - Playing: position-aware (leading vs following vs last)
    - Tracking: monitors played cards to make informed decisions
    - Defense: detects and blocks shoot-the-moon attempts

    All heuristics use configurable weights (see AIWeights class).
    """

    QUEEN_OF_SPADES = Card(Suit.SPADES, Rank.QUEEN)

    def __init__(self, weights: AIWeights = None):
        self.weights = weights or DEFAULT_WEIGHTS[GameMode.PLAYER_4]
        self.tracker = CardTracker()
        self.points_taken: dict[int, int] = {}  # player_index -> points this round

    def reset_round(self, num_players: int):
        """Reset tracking for a new round."""
        self.tracker.reset()
        self.points_taken = {i: 0 for i in range(num_players)}

    def record_trick(
        self,
        trick_cards: list[Card],
        winner_index: int,
        trick_info: list[tuple[int, Card]] = None,
    ):
        """
        Record a completed trick.

        Args:
            trick_cards: List of cards played in the trick.
            winner_index: Index of the player who won the trick.
            trick_info: Optional list of (player_index, card) tuples for advanced tracking.
        """
        self.tracker.record_cards(trick_cards)

        # Track voids if trick_info is provided
        if trick_info:
            lead_suit = trick_info[0][1].suit
            for player_idx, card in trick_info:
                if card.suit != lead_suit:
                    self.tracker.mark_player_void(player_idx, lead_suit)

        points = sum(c.points for c in trick_cards)
        self.points_taken[winner_index] = (
            self.points_taken.get(winner_index, 0) + points
        )

    # =========================================================================
    # PASSING STRATEGY
    # =========================================================================

    def select_pass_cards(self, hand: list[Card], direction: str) -> list[Card]:
        """
        Select 3 cards to pass.

        Strategy (using configurable weights):
        1. Pass Queen of Spades if not well-protected
        2. Try to void a short suit (not spades if holding Q♠)
        3. Pass high hearts
        4. Pass high cards (Aces, Kings) that could win unwanted tricks
        """
        candidates = []
        hand_set = set(hand)
        w = self.weights  # Shorthand

        # Analyze hand
        suits_count = {suit: [] for suit in Suit}
        for card in hand:
            suits_count[card.suit].append(card)

        for suit in suits_count:
            suits_count[suit].sort(key=lambda c: c.rank.value)

        has_queen_spades = self.QUEEN_OF_SPADES in hand_set

        # Determine directional bonuses/penalties
        is_pass_left = direction == "LEFT"
        is_pass_right = direction == "RIGHT"
        is_pass_across = direction == "ACROSS"

        # 1. Pass Queen of Spades if dangerous
        if has_queen_spades:
            spades = suits_count[Suit.SPADES]
            # Q♠ is dangerous if we have few low spades to protect it
            low_spades = [c for c in spades if c.rank.value < Rank.QUEEN.value]
            if len(low_spades) < w.queen_protection_threshold:
                priority = w.pass_queen_of_spades
                if is_pass_across:
                    priority += w.pass_across_danger_bonus
                candidates.append((self.QUEEN_OF_SPADES, priority))

        # 2. Try to void a short suit
        # Find shortest non-spades suit (or spades if no Q♠)
        voidable_suits = []
        for suit, cards in suits_count.items():
            if suit == Suit.SPADES and has_queen_spades:
                continue  # Don't void spades if we have Q♠
            if 1 <= len(cards) <= 3:
                voidable_suits.append((suit, len(cards), cards))

        # Prioritize voiding diamonds or clubs (less dangerous)
        voidable_suits.sort(
            key=lambda x: (
                x[1],  # Fewer cards = easier to void
                0 if x[0] in (Suit.DIAMONDS, Suit.CLUBS) else 1,  # Prefer minor suits
            )
        )

        if voidable_suits:
            suit, count, cards = voidable_suits[0]
            for card in cards:
                # Higher priority for voiding, bonus for minor suits
                priority = w.pass_to_void_suit + card.rank.value
                if suit in (Suit.DIAMONDS, Suit.CLUBS):
                    priority += w.void_minor_suit_bonus

                if is_pass_right:
                    priority += w.pass_right_void_bonus

                candidates.append((card, priority))

        # 3. Pass high hearts
        for card in suits_count[Suit.HEARTS]:
            if card.rank.value >= w.high_heart_threshold:
                priority = w.pass_high_hearts + card.rank.value
                if is_pass_left:
                    priority -= w.pass_left_high_card_penalty
                if is_pass_across:
                    priority += w.pass_across_danger_bonus
                if card not in [c for c, _ in candidates]:
                    candidates.append((card, priority))

        # 4. Pass other high cards (Aces, Kings, Queens)
        for suit in [Suit.CLUBS, Suit.DIAMONDS, Suit.SPADES]:
            for card in suits_count[suit]:
                if card.rank.value >= w.high_card_threshold:
                    if card not in [c for c, _ in candidates]:
                        priority = w.pass_high_cards + card.rank.value
                        if is_pass_left:
                            priority -= w.pass_left_high_card_penalty
                        # Special case for A/K Spades when dangerous
                        if (
                            card.suit == Suit.SPADES
                            and card.rank.value > Rank.QUEEN.value
                            and not has_queen_spades
                        ):
                            priority += w.pass_dangerous_ak_spades
                        candidates.append((card, priority))

        # 5. Strategic 2 of Clubs and Ace of Clubs control
        two_of_clubs = Card(Suit.CLUBS, Rank.TWO)
        ace_of_clubs = Card(Suit.CLUBS, Rank.ACE)

        if two_of_clubs in hand_set:
            # Sometimes pass 2 of clubs to control who starts
            priority = w.pass_two_of_clubs_control
            if ace_of_clubs in hand_set:
                # If we have Ace, keeping 2 is better (retain Ace bonus)
                priority -= w.retain_ace_of_clubs_bonus
            candidates.append((two_of_clubs, priority))

        # 6. Decoy passes (mix high and low)
        # If we already have 2 high cards, consider a low card as a decoy
        high_candidates = [c for c, p in candidates if p > 50]
        if len(high_candidates) >= 2:
            low_cards = sorted(hand, key=lambda c: c.rank.value)
            if low_cards:
                candidates.append((low_cards[0], w.pass_decoy_bonus))

        # 7. Penalty for passing the last card of a suit (signals void)
        for card, priority in candidates:
            suit_cards = suits_count[card.suit]
            if len(suit_cards) == 1:
                # This is the last card of this suit in our hand
                # Update priority in candidates list
                for i, (c, p) in enumerate(candidates):
                    if c == card:
                        candidates[i] = (c, p - w.pass_last_card_penalty)
                        break

        # 8. Fill remaining with highest cards
        all_cards_by_rank = sorted(hand, key=lambda c: c.rank.value, reverse=True)
        for card in all_cards_by_rank:
            if card not in [c for c, _ in candidates]:
                candidates.append((card, w.pass_base_priority + card.rank.value))

        # Sort by priority and take top 3 unique cards
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = []
        seen = set()
        for card, _ in candidates:
            if card not in seen and len(chosen) < 3:
                chosen.append(card)
                seen.add(card)

        # Record what we passed (Internal tracking)
        # We don't know the exact absolute index of the destination yet,
        # but we can store it by direction for later resolution if needed.
        # For now, just return the cards.
        return chosen

    # =========================================================================
    # PLAYING STRATEGY
    # =========================================================================

    def select_play(
        self,
        hand: list[Card],
        valid_plays: list[Card],
        trick: Trick,
        player_index: int,
        num_players: int,
        hearts_broken: bool,
    ) -> Card:
        """
        Select a card to play.

        Strategy varies by position:
        - Leading: careful suit selection
        - Following (not last): try to duck or dump
        - Last to play: take only if safe, otherwise dump highest
        """
        if not valid_plays:
            raise ValueError("No valid plays")

        if len(valid_plays) == 1:
            return valid_plays[0]

        # Determine position in trick
        cards_in_trick = len(trick.cards)
        is_leading = cards_in_trick == 0
        is_last = cards_in_trick == num_players - 1

        # Check for shoot-the-moon threat
        moon_threat = self._detect_moon_threat(player_index, num_players)

        if is_leading:
            return self._select_lead(
                hand, valid_plays, hearts_broken, moon_threat, player_index, num_players
            )
        else:
            return self._select_follow(
                hand,
                valid_plays,
                trick,
                is_last,
                moon_threat,
                player_index,
                num_players,
            )

    def _should_attempt_moon(self, hand: list[Card]) -> bool:
        """Evaluate if we should attempt to shoot the moon."""
        w = self.weights
        if not w.moon_attempt_threshold:
            return False

        strength = 0
        suits_count = {suit: [] for suit in Suit}
        for card in hand:
            suits_count[card.suit].append(card)

        # High hearts are good for moon
        high_hearts = [
            c for c in suits_count[Suit.HEARTS] if c.rank.value >= Rank.TEN.value
        ]
        strength += len(high_hearts) * w.moon_high_hearts_weight

        # High spades are critical
        if self.QUEEN_OF_SPADES in hand:
            strength += w.moon_queen_spades_weight

        high_spades = [
            c for c in suits_count[Suit.SPADES] if c.rank.value >= Rank.KING.value
        ]
        strength += len(high_spades) * w.moon_high_spades_weight

        # High cards in other suits
        for suit in [Suit.CLUBS, Suit.DIAMONDS]:
            high_cards = [
                c for c in suits_count[suit] if c.rank.value >= Rank.JACK.value
            ]
            strength += len(high_cards) * 5  # Implicit weight for other high cards

        return strength >= w.moon_attempt_threshold

    def _detect_moon_threat(self, my_index: int, num_players: int) -> int | None:
        """
        Detect if someone might be shooting the moon.
        Returns the player index if there's a threat, None otherwise.
        Uses weights.moon_threat_threshold to determine sensitivity.
        """
        for idx, points in self.points_taken.items():
            if idx == my_index:
                continue
            # If someone has taken significant points and no one else has any
            if points >= self.weights.moon_threat_threshold:
                others_points = sum(
                    p
                    for i, p in self.points_taken.items()
                    if i != idx and i != my_index
                )
                my_points = self.points_taken.get(my_index, 0)
                if others_points == 0 and my_points == 0:
                    return idx
        return None

    def _select_lead(
        self,
        hand: list[Card],
        valid_plays: list[Card],
        hearts_broken: bool,
        moon_threat: int | None,
        my_index: int,
        num_players: int,
    ) -> Card:
        """Select a card when leading a trick using weighted priorities."""
        w = self.weights

        # Group valid plays by suit
        by_suit: dict[Suit, list[Card]] = {suit: [] for suit in Suit}
        for card in valid_plays:
            by_suit[card.suit].append(card)

        for suit in by_suit:
            by_suit[suit].sort(key=lambda c: c.rank.value)

        # Check if we are shooting the moon
        is_shooting_moon = self._should_attempt_moon(hand)

        # If someone else is shooting the moon, try to take points
        if moon_threat is not None:
            # Lead hearts to spread points
            if by_suit[Suit.HEARTS]:
                return max(by_suit[Suit.HEARTS], key=lambda c: c.rank.value)

        # If WE are shooting moon, lead high cards to take control
        if is_shooting_moon:
            # Lead highest card overall if it's likely to win
            return max(valid_plays, key=lambda c: c.rank.value)

        # Calculate weighted priority for each valid play
        queen_gone = self.tracker.queen_of_spades_played()

        # Suit base priorities
        suit_priorities = {
            Suit.CLUBS: w.lead_clubs_priority,
            Suit.DIAMONDS: w.lead_diamonds_priority,
            Suit.SPADES: w.lead_spades_priority if queen_gone else 0,
            Suit.HEARTS: w.lead_hearts_priority if hearts_broken else 0,
        }

        best_card = None
        best_score = -float("inf")

        for card in valid_plays:
            # Base score from suit priority
            score = suit_priorities[card.suit]

            # Prefer low cards (subtract rank * preference multiplier)
            score -= card.rank.value * w.lead_low_card_preference

            # 1. Queen flushing strategy & Bleeding Spades
            if not queen_gone and card.suit == Suit.SPADES:
                spades_in_hand = by_suit[Suit.SPADES]
                if self.QUEEN_OF_SPADES in hand:
                    # Bleeding Spades: If we have Q and protection, force others to play spades
                    if len(spades_in_hand) >= w.queen_protection_threshold:
                        score += w.bleed_spades_priority
                elif card.rank.value <= w.flush_queen_max_rank:
                    if len(spades_in_hand) >= w.flush_queen_min_spades:
                        score += w.flush_queen_priority

            # 2. Avoid void opponents
            for other_player in range(num_players):
                if other_player == my_index:
                    continue

                if self.tracker.is_player_void(other_player, card.suit):
                    # Penalize lead if a dangerous opponent is void
                    score -= w.lead_avoid_void_opponent

                    # Extra bonus if we know the Queen holder is void in this suit
                    if not queen_gone and self.QUEEN_OF_SPADES not in hand:
                        # We don't know who has it, but if someone is void, it's risky
                        pass

            if score > best_score:
                best_score = score
                best_card = card

        return best_card if best_card else min(valid_plays, key=lambda c: c.rank.value)

    def _select_follow(
        self,
        hand: list[Card],
        valid_plays: list[Card],
        trick: Trick,
        is_last: bool,
        moon_threat: int | None,
        my_index: int,
        num_players: int,
    ) -> Card:
        """Select a card when following in a trick using weighted preferences."""
        w = self.weights

        lead_suit = trick.lead_suit
        trick_cards = [card for _, card in trick.cards]

        # Check if we are following suit or discarding
        following_suit = valid_plays[0].suit == lead_suit

        is_shooting_moon = self._should_attempt_moon(hand)

        if not following_suit:
            # We can't follow suit - opportunity to dump!
            return self._select_discard(
                hand, valid_plays, moon_threat, is_shooting_moon
            )

        # Following suit - analyze the trick
        lead_suit_cards_in_trick = [c for c in trick_cards if c.suit == lead_suit]
        highest_in_trick = max(lead_suit_cards_in_trick, key=lambda c: c.rank.value)

        # Separate our plays into those that win and those that duck
        winning_plays = [
            c for c in valid_plays if c.rank.value > highest_in_trick.rank.value
        ]
        ducking_plays = [
            c for c in valid_plays if c.rank.value < highest_in_trick.rank.value
        ]

        # Finesse logic: playing mid-rank to draw high cards from others
        # (This is most useful when we are NOT last and someone after us might have a point card)
        if not is_last and ducking_plays:
            # If we don't have the highest card overall in this suit (tracked),
            # consider a high-duck as a finesse.
            remaining_higher = [
                c
                for c in self.tracker.remaining_in_suit(lead_suit)
                if c.rank.value
                > max(valid_plays, key=lambda x: x.rank.value).rank.value
            ]
            if remaining_higher:
                # We can't win anyway if they play high, so try to draw it out with a mid-rank duck
                # This is already somewhat covered by high_duck_preference, but let's boost it.
                finesse_candidates = [
                    c
                    for c in ducking_plays
                    if Rank.NINE.value <= c.rank.value <= Rank.JACK.value
                ]
                if finesse_candidates:
                    # Use highest finesse card (which is a duck)
                    return max(
                        finesse_candidates,
                        key=lambda c: c.rank.value * w.high_duck_preference,
                    )

        trick_has_points = any(c.points > 0 for c in trick_cards)

        if moon_threat is not None and trick_has_points:
            if winning_plays:
                # Win cheaply to block
                return min(
                    winning_plays, key=lambda c: c.rank.value * w.low_win_preference
                )

        # If WE are shooting moon, try to win points
        if is_shooting_moon and trick_has_points:
            if winning_plays:
                # Win with highest card to keep control
                return max(winning_plays, key=lambda c: c.rank.value)

        if is_last:
            # Last to play - we know exactly what we'd win
            if not trick_has_points:
                # No points in trick - safe to win, play highest (weighted)
                return max(
                    valid_plays,
                    key=lambda c: c.rank.value * w.take_safe_trick_preference,
                )
            else:
                # Points in trick - duck if possible
                if ducking_plays:
                    # High duck (weighted)
                    return max(
                        ducking_plays,
                        key=lambda c: c.rank.value * w.high_duck_preference,
                    )
                else:
                    # Must win - play lowest winning card (weighted)
                    return min(
                        winning_plays, key=lambda c: c.rank.value * w.low_win_preference
                    )
        else:
            # Not last - someone plays after us
            if ducking_plays:
                # Play high duck to save low cards (weighted)
                return max(
                    ducking_plays, key=lambda c: c.rank.value * w.high_duck_preference
                )
            else:
                # We must win - play lowest winner and hope (weighted)
                # But wait, preserve exit cards if possible
                exit_cards = [
                    c for c in winning_plays if c.rank.value <= w.exit_card_threshold
                ]
                non_exit_winners = [
                    c for c in winning_plays if c.rank.value > w.exit_card_threshold
                ]

                if non_exit_winners:
                    return min(
                        non_exit_winners,
                        key=lambda c: c.rank.value * w.low_win_preference,
                    )
                return min(
                    winning_plays, key=lambda c: c.rank.value * w.low_win_preference
                )

    def _select_discard(
        self,
        hand: list[Card],
        valid_plays: list[Card],
        moon_threat: int | None,
        is_shooting_moon: bool = False,
    ) -> Card:
        """Select a card to discard when we can't follow suit using weighted priorities."""
        w = self.weights

        # If we are shooting moon, keep high cards and points
        if is_shooting_moon:
            # Discard non-point cards, preferably low cards that aren't useful as winners
            # Actually, in discard, we want to get rid of cards that might lose us control later,
            # but we also want to keep cards that WIN points.
            # But wait, discarding a point card means someone ELSE gets it. That fails the moon.
            # So we MUST NOT discard hearts or the Queen.
            safe_discards = [c for c in valid_plays if c.points == 0]
            if safe_discards:
                # Discard highest non-point card that isn't a likely winner
                # But wait: if we HAVE control (lots of high cards), dump low hearts
                # to ensure we win the POINTY tricks later.
                if self._has_suit_control(hand, Suit.HEARTS):
                    low_hearts = sorted(
                        [c for c in valid_plays if c.suit == Suit.HEARTS],
                        key=lambda c: c.rank.value,
                    )
                    if low_hearts:
                        return low_hearts[0]
                return max(safe_discards, key=lambda c: c.rank.value)

        # If blocking a moon shooter, keep points (discard non-point cards)
        if moon_threat is not None:
            non_point_cards = [c for c in valid_plays if c.points == 0]
            if non_point_cards:
                return max(non_point_cards, key=lambda c: c.rank.value)

        # Calculate discard priority for each card
        best_card = None
        best_score = -float("inf")

        queen_gone = self.tracker.queen_of_spades_played()

        for card in valid_plays:
            score = 0

            # Queen of Spades - highest priority to dump
            if card == self.QUEEN_OF_SPADES:
                score = w.discard_queen_of_spades

            # Hearts - high priority to dump
            elif card.suit == Suit.HEARTS:
                score = w.discard_hearts + card.rank.value * w.discard_rank_multiplier

            # Dangerous spades (A, K when Q♠ not played)
            elif (
                card.suit == Suit.SPADES
                and card.rank.value > Rank.QUEEN.value
                and not queen_gone
            ):
                score = (
                    w.discard_dangerous_spades
                    + card.rank.value * w.discard_rank_multiplier
                )

            # Other high cards
            else:
                score = (
                    w.discard_high_cards + card.rank.value * w.discard_rank_multiplier
                )

            if score > best_score:
                best_score = score
                best_card = card

        return best_card if best_card else max(valid_plays, key=lambda c: c.rank.value)

    def _has_suit_control(self, hand: list[Card], suit: Suit) -> bool:
        """Check if we have enough high cards in a suit to control it."""
        w = self.weights
        high_cards = [
            c for c in hand if c.suit == suit and c.rank.value >= Rank.JACK.value
        ]
        return len(high_cards) >= w.control_card_count


# Convenience functions for simpler integration
_ai_instances: dict[int, HeartsAI] = {}
_ai_weights: AIWeights = DEFAULT_WEIGHTS


def set_ai_weights(weights: AIWeights):
    """Set the weights to use for all AI instances."""
    global _ai_weights
    _ai_weights = weights


def get_ai_weights() -> AIWeights:
    """Get the current AI weights."""
    return _ai_weights


def get_ai(player_index: int) -> HeartsAI:
    """Get or create an AI instance for a player."""
    if player_index not in _ai_instances:
        _ai_instances[player_index] = HeartsAI(weights=_ai_weights)
    return _ai_instances[player_index]


def reset_all_ai(
    num_players: int, mode: GameMode = GameMode.PLAYER_4, weights: AIWeights = None
):
    """Reset all AI instances for a new round."""
    global _ai_weights, _ai_mode
    _ai_mode = mode

    if weights is not None:
        _ai_weights = weights
    else:
        # Use existing weights dictionary if available, otherwise default
        _ai_weights = DEFAULT_WEIGHTS.get(mode, AIWeights())

    _ai_instances.clear()
    for i in range(num_players):
        _ai_instances[i] = HeartsAI(weights=_ai_weights)
        _ai_instances[i].reset_round(num_players)


def record_trick_for_all(
    trick_cards: list[Card],
    winner_index: int,
    trick_info: list[tuple[int, Card]] = None,
):
    """Record a trick completion for all AI instances."""
    for ai in _ai_instances.values():
        ai.record_trick(trick_cards, winner_index, trick_info)


def get_weights_summary() -> str:
    """Get a formatted summary of all AI weights and their values."""
    w = _ai_weights
    return f"""
AI HEURISTIC WEIGHTS (Mode: {_ai_mode.name})
====================

PASSING STRATEGY
----------------
pass_queen_of_spades:      {w.pass_queen_of_spades:6.1f}  (Priority to pass QoS when unprotected)
queen_protection_threshold:    {w.queen_protection_threshold:3d}  (Low spades needed to keep QoS)
pass_to_void_suit:         {w.pass_to_void_suit:6.1f}  (Priority to void a short suit)
void_minor_suit_bonus:     {w.void_minor_suit_bonus:6.1f}  (Extra priority for voiding clubs/diamonds)
pass_high_hearts:          {w.pass_high_hearts:6.1f}  (Priority to pass high hearts)
high_heart_threshold:          {w.high_heart_threshold:3d}  (Min rank for "high" heart, 10=Ten)
pass_high_cards:           {w.pass_high_cards:6.1f}  (Priority to pass A/K/Q of other suits)
high_card_threshold:           {w.high_card_threshold:3d}  (Min rank for "high" card, 12=Queen)
pass_base_priority:        {w.pass_base_priority:6.1f}  (Base priority + rank for any card)

LEADING STRATEGY
----------------
lead_clubs_priority:       {w.lead_clubs_priority:6.1f}  (Priority to lead clubs)
lead_diamonds_priority:    {w.lead_diamonds_priority:6.1f}  (Priority to lead diamonds)
lead_spades_priority:      {w.lead_spades_priority:6.1f}  (Priority to lead spades after QoS gone)
lead_hearts_priority:      {w.lead_hearts_priority:6.1f}  (Priority to lead hearts after broken)
lead_low_card_preference:  {w.lead_low_card_preference:6.1f}  (Multiplier favoring low cards)

DISCARDING STRATEGY
-------------------
discard_queen_of_spades:   {w.discard_queen_of_spades:6.1f}  (Priority to dump QoS)
discard_hearts:            {w.discard_hearts:6.1f}  (Priority to dump hearts)
discard_dangerous_spades:  {w.discard_dangerous_spades:6.1f}  (Priority to dump A/K spades)
discard_high_cards:        {w.discard_high_cards:6.1f}  (Priority to dump other high cards)
discard_rank_multiplier:   {w.discard_rank_multiplier:6.1f}  (Rank contribution to discard score)

FOLLOWING SUIT STRATEGY
-----------------------
high_duck_preference:      {w.high_duck_preference:6.1f}  (Prefer high cards when ducking)
low_win_preference:        {w.low_win_preference:6.1f}  (Prefer low cards when must win)
take_safe_trick_preference:{w.take_safe_trick_preference:6.1f}  (Preference for taking point-free tricks)

MOON DEFENSE
------------
moon_threat_threshold:         {w.moon_threat_threshold:3d}  (Points to suspect moon attempt)
moon_block_priority:       {w.moon_block_priority:6.1f}  (Priority boost when blocking)
"""

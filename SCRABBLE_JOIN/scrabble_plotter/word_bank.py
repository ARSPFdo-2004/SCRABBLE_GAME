from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable


REFERENCE_WORD_LIMIT = 1000
HORIZONTAL_LEFT_TO_RIGHT = "horizontal_left_to_right"
VERTICAL_TOP_TO_BOTTOM = "vertical_top_to_bottom"
ALLOWED_DIRECTIONS = {HORIZONTAL_LEFT_TO_RIGHT, VERTICAL_TOP_TO_BOTTOM}
DIRECTION_LABELS = {
    HORIZONTAL_LEFT_TO_RIGHT: "Horizontal left-to-right",
    VERTICAL_TOP_TO_BOTTOM: "Vertical top-to-bottom",
}
DIRECTION_ORDER = (HORIZONTAL_LEFT_TO_RIGHT, VERTICAL_TOP_TO_BOTTOM)


@dataclass(frozen=True)
class MatrixWord:
    word: str
    direction: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int

    @property
    def direction_label(self) -> str:
        return DIRECTION_LABELS.get(self.direction, self.direction)

    @property
    def start_cell(self) -> str:
        return _cell_label(self.start_row, self.start_col)

    @property
    def end_cell(self) -> str:
        return _cell_label(self.end_row, self.end_col)


def normalize_word(value: Any) -> str:
    return "".join(char for char in str(value).strip().upper() if "A" <= char <= "Z")


@lru_cache(maxsize=1)
def generated_reference_words() -> tuple[str, ...]:
    words: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        word = normalize_word(value)
        if len(word) < 2 or word in seen:
            return
        seen.add(word)
        words.append(word)

    for word in _CORE_WORDS:
        add(word)

    for noun in _NOUN_ROOTS:
        add(noun)

    for verb in _REGULAR_VERBS:
        add(verb)

    for adjective in _ADJECTIVES:
        add(adjective)

    for noun in _NOUN_ROOTS:
        add(_plural(noun))

    for verb in _REGULAR_VERBS:
        add(_third_person(verb))
        add(_past_tense(verb))
        add(_present_participle(verb))

    for adjective in _ADJECTIVES:
        add(_adverb(adjective))
        add(_comparative(adjective))
        add(_superlative(adjective))

    for prefix in ("RE", "UN", "MIS", "OVER", "UNDER"):
        for verb in _PREFIXABLE_VERBS:
            add(prefix + verb)

    if len(words) < REFERENCE_WORD_LIMIT:
        raise RuntimeError(
            f"Reference word generator produced {len(words)} words; "
            f"{REFERENCE_WORD_LIMIT} are required."
        )
    return tuple(words[:REFERENCE_WORD_LIMIT])


@lru_cache(maxsize=1)
def generated_reference_word_set() -> frozenset[str]:
    return frozenset(generated_reference_words())


def word_matches_reference(word: Any) -> bool:
    return normalize_word(word) in generated_reference_word_set()


def filter_matching_words(words: Iterable[Any]) -> list[Any]:
    matches: list[Any] = []
    bank = generated_reference_word_set()
    for item in words:
        word = normalize_word(getattr(item, "word", ""))
        direction = str(getattr(item, "direction", "")).strip().lower()
        if direction in ALLOWED_DIRECTIONS and word in bank:
            matches.append(item)
    return matches


def identify_matrix_words(
    matrix: list[list[str]],
    min_word_length: int = 2,
) -> list[MatrixWord]:
    words: list[MatrixWord] = []
    if not matrix:
        return words

    row_count = len(matrix)
    col_count = max((len(row) for row in matrix), default=0)
    if col_count == 0:
        return words

    for row_index, row in enumerate(matrix):
        run: list[tuple[int, str]] = []
        for col_index in range(col_count):
            letter = _matrix_letter(row[col_index] if col_index < len(row) else "")
            if letter:
                run.append((col_index, letter))
            else:
                words.extend(_matrix_run_to_word(run, row_index, HORIZONTAL_LEFT_TO_RIGHT, min_word_length))
                run = []
        words.extend(_matrix_run_to_word(run, row_index, HORIZONTAL_LEFT_TO_RIGHT, min_word_length))

    for col_index in range(col_count):
        run: list[tuple[int, str]] = []
        for row_index in range(row_count):
            row = matrix[row_index]
            letter = _matrix_letter(row[col_index] if col_index < len(row) else "")
            if letter:
                run.append((row_index, letter))
            else:
                words.extend(_matrix_run_to_word(run, col_index, VERTICAL_TOP_TO_BOTTOM, min_word_length))
                run = []
        words.extend(_matrix_run_to_word(run, col_index, VERTICAL_TOP_TO_BOTTOM, min_word_length))

    return words


def matched_matrix_words(
    matrix: list[list[str]],
    min_word_length: int = 2,
) -> list[MatrixWord]:
    return filter_matching_words(identify_matrix_words(matrix, min_word_length=min_word_length))


def format_words_by_direction(words: Iterable[Any]) -> str:
    grouped = {direction: [] for direction in DIRECTION_ORDER}
    for word in words:
        direction = str(getattr(word, "direction", "")).strip().lower()
        if direction in grouped:
            grouped[direction].append(word)

    lines: list[str] = []
    for direction in DIRECTION_ORDER:
        direction_words = grouped[direction]
        if not direction_words:
            continue
        if lines:
            lines.append("")
        lines.append(f"{DIRECTION_LABELS[direction]}:")
        for index, detected in enumerate(direction_words, start=1):
            lines.append(f"{index}. {_format_word(detected)}")

    return "\n".join(lines) if lines else "No matching words found."


def _format_word(detected: Any) -> str:
    word = normalize_word(getattr(detected, "word", ""))
    details: list[str] = []

    confidence = getattr(detected, "confidence", None)
    if isinstance(confidence, (int, float)) and confidence > 0:
        details.append(f"{confidence:.0f}%")

    start_cell = getattr(detected, "start_cell", "")
    end_cell = getattr(detected, "end_cell", "")
    if start_cell and end_cell:
        details.append(f"{start_cell}-{end_cell}")

    return f"{word} ({', '.join(details)})" if details else word


def _matrix_run_to_word(
    run: list[tuple[int, str]],
    fixed_index: int,
    direction: str,
    min_word_length: int,
) -> list[MatrixWord]:
    if len(run) < min_word_length:
        return []
    word = "".join(letter for _, letter in run)
    start_index = run[0][0]
    end_index = run[-1][0]
    if direction == HORIZONTAL_LEFT_TO_RIGHT:
        return [
            MatrixWord(
                word=word,
                direction=direction,
                start_row=fixed_index,
                start_col=start_index,
                end_row=fixed_index,
                end_col=end_index,
            )
        ]
    return [
        MatrixWord(
            word=word,
            direction=direction,
            start_row=start_index,
            start_col=fixed_index,
            end_row=end_index,
            end_col=fixed_index,
        )
    ]


def _matrix_letter(value: Any) -> str:
    word = normalize_word(value)
    return word if len(word) == 1 else ""


def _cell_label(row: int, col: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def _plural(word: str) -> str:
    if word.endswith(("S", "X", "Z", "CH", "SH")):
        return word + "ES"
    if len(word) > 1 and word.endswith("Y") and word[-2] not in _VOWELS:
        return word[:-1] + "IES"
    return word + "S"


def _third_person(word: str) -> str:
    if word.endswith(("S", "X", "Z", "CH", "SH", "O")):
        return word + "ES"
    if len(word) > 1 and word.endswith("Y") and word[-2] not in _VOWELS:
        return word[:-1] + "IES"
    return word + "S"


def _past_tense(word: str) -> str:
    if word.endswith("E"):
        return word + "D"
    if len(word) > 1 and word.endswith("Y") and word[-2] not in _VOWELS:
        return word[:-1] + "IED"
    if _ends_cvc(word):
        return word + word[-1] + "ED"
    return word + "ED"


def _present_participle(word: str) -> str:
    if word.endswith("IE"):
        return word[:-2] + "YING"
    if word.endswith("E") and not word.endswith(("EE", "YE")):
        return word[:-1] + "ING"
    if _ends_cvc(word):
        return word + word[-1] + "ING"
    return word + "ING"


def _adverb(word: str) -> str:
    if word.endswith("Y") and len(word) > 1 and word[-2] not in _VOWELS:
        return word[:-1] + "ILY"
    if word.endswith("LE"):
        return word[:-1] + "Y"
    return word + "LY"


def _comparative(word: str) -> str:
    if word.endswith("E"):
        return word + "R"
    if word.endswith("Y") and len(word) > 1 and word[-2] not in _VOWELS:
        return word[:-1] + "IER"
    if _ends_cvc(word):
        return word + word[-1] + "ER"
    return word + "ER"


def _superlative(word: str) -> str:
    if word.endswith("E"):
        return word + "ST"
    if word.endswith("Y") and len(word) > 1 and word[-2] not in _VOWELS:
        return word[:-1] + "IEST"
    if _ends_cvc(word):
        return word + word[-1] + "EST"
    return word + "EST"


def _ends_cvc(word: str) -> bool:
    if len(word) < 3 or word[-1] in _VOWELS or word[-1] in {"W", "X", "Y"}:
        return False
    return word[-2] in _VOWELS and word[-3] not in _VOWELS


_VOWELS = {"A", "E", "I", "O", "U"}

_CORE_WORDS = """
AA AB AD AE AG AH AI AL AM AN AR AS AT AW AX AY BA BE BI BO BY DA DE DO ED EF EH EL EM EN ER ES ET EW EX FA FE GI GO
HA HE HI HO ID IF IN IS IT JO KA KI KO LA LI LO MA ME MI MM MO MU MY NA NE NO NU OD OE OF OH OI OM ON OP OR OS OW OX
OY PA PE PI QI RE SH SI SO TA TE TI TO UH UM UN UP US UT WE WO XI XU YA YE YO ZA
ACE ACT AGE AGO AID AIM AIR ALE ALL AND ANT ANY APE ARC ARE ARM ART ASH ATE AWE BAD BAG BAN BAR BAT BAY BED BEE BET BIG
BIN BIT BOW BOX BOY BUN BUS BUT BUY CAB CAN CAP CAR CAT COW CRY CUP CUT DAY DEN DID DIE DIG DIM DIP DOG DOT DRY DUE EAR
EAT EGG END EYE FAR FAT FED FEE FEW FIG FIN FIR FIT FIX FLY FOG FOR FOX FRY FUN GAP GAS GEM GET GIN GOT GUM GUN GUY HAD
HAM HAS HAT HAY HEN HER HID HIM HIP HIS HIT HOT HOW HUG ICE ILL INK JAM JAR JET JOB JOY KEY KID KIT LAB LAD LAG LAP LAW
LAY LED LEG LET LID LIE LIP LOG LOT LOW MAD MAN MAP MAT MAY MEN MET MIX MOP MUD MUG NET NEW NOD NOR NOT NOW NUT OAK OAR
ODD OFF OIL OLD ONE OUR OUT OWL OWN PAD PAL PAN PAT PAW PAY PEN PET PIE PIN PIT POT RAG RAM RAN RAT RAW RED RID RIG RIM
RIP ROB ROD ROW RUB RUG RUN SAD SAT SAW SAY SEA SEE SET SHE SHY SIN SIP SIR SIT SKY SLY SON SOW SOY SUN TAB TAG TAN TAP
TAR TAX TEA TEN THE TIE TIN TIP TOE TON TOP TOY TRY TWO USE VAN VAT VET WAR WAS WAY WEB WED WET WHO WHY WIN WIT YES YET
YOU ZIP ZOO
ABLE ABOUT ABOVE ACID AFTER AGAIN ALBUM ALIVE ALONE ALONG ALSO AMONG ANGEL ANGER APPLE APRIL AREA ARENA ARMY AUDIO BASIC
BEACH BEAR BEAT BEAUTY BIRD BIRTH BLACK BLUE BOARD BRAIN BREAD BREAK BRIDGE BRIGHT BROWN BUILD CAMP CARD CARE CASE CHAIR
CHANCE CHANGE CHILD CITY CLASS CLEAN CLEAR CLOCK CLOUD COACH COAST COLOR COURT COVER CREAM DANCE DARK DATA DEAL DEATH
DEEP DESK DOOR DREAM DRESS DRIVE EARTH EAST EDGE ENERGY EVEN EVENT EVERY FACE FACT FALL FAMILY FARM FAST FATHER FIELD FIRE
FISH FLOOR FLOWER FOOD FORCE FORM FRIEND GAME GARDEN GIRL GLASS GOLD GOOD GRASS GREEN GROUP HAND HAPPY HARD HEART HELP
HOME HORSE HOTEL HOUR HOUSE IDEA IMAGE IRON ISLAND JOB KIND KING LAND LARGE LATE LIFE LIGHT LINE LIST LONG LOVE MACHINE
MARCH MARKET MEAL MEAN MONEY MONTH MORNING MOTHER MUSIC NAME NIGHT NORTH NOTE OFFICE OPEN ORANGE ORDER PAGE PAPER PARTY
PEACE PHONE PLACE PLANE PLANT POINT POWER PRICE QUICK RADIO RAIN READY RIVER ROAD ROOM SCHOOL SCORE SEA SHAPE SHORT SIDE
SIMPLE SMALL SOUND SOUTH SPACE SPEED SPRING STAR STORY STREET TABLE TEAM THING TIME TRAIN TREE TRUE VALUE VIDEO WATER WEEK
WEST WHITE WIND WOMAN WORD WORLD YEAR
""".split()

_NOUN_ROOTS = """
ACTOR AIRPORT ANIMAL ANSWER ARTIST BABY BALL BANK BATTLE BEACH BELL BIKE BIRD BIRTH BLOCK BOOK BOTTLE BOWL BOX BRANCH
BREAD BRUSH BUTTON CAMERA CAMP CANDLE CANDY CARD CASTLE CHAIR CHILD CHURCH CITY CLASS CLOCK CLOUD COACH COAT COIN CORN
COUNTRY COURT CROWN CUP DESK DOCTOR DOOR DREAM DRESS DRIVER EARTH EDGE ENGINE FACE FACT FARM FIELD FILE FIRE FISH FLOOR
FLOWER FOOD FOREST FRIEND GAME GARDEN GATE GIRL GLASS GLOVE GOLD GROUP GUIDE HALL HAND HAT HEART HILL HOME HORSE HOTEL
HOUSE ISLAND KING KITCHEN LAKE LAMP LAND LEAF LETTER LINE MARKET MEAL MONEY MONTH MOTHER MOUNTAIN MUSIC NAME NIGHT NOTE
OCEAN OFFICE PAGE PAPER PARK PARTY PEN PHONE PICTURE PLANE PLANT PLAYER PRICE QUEEN RAIN RIVER ROAD ROCK ROOM SCHOOL
SCREEN SEAT SHAPE SHIP SHOE SHOP SONG SOUND SPACE STAR STONE STORY STREET STUDENT TABLE TEACHER TEAM TICKET TRAIN TREE
VALUE VIDEO WALL WATCH WATER WEEK WINDOW WOMAN WORD WORLD YEAR
""".split()

_REGULAR_VERBS = """
ACCEPT ADD ADMIT ADOPT ADVISE AGREE ALLOW ANSWER APPEAR APPLY ARGUE ARRIVE ASK ATTACK BAKE BALANCE BATH CALL CARE CARRY
CAUSE CHANGE CHECK CLEAN CLEAR CLOSE COLLECT COMBINE COMPARE COMPLAIN COMPLETE CONNECT CONSIDER CONTINUE COOK COPY COUNT
COVER CREATE CROSS CRY DANCE DECIDE DELIVER DENY DESCRIBE DESIGN DESTROY DEVELOP DIE DISCOVER DISCUSS DIVIDE DROP EARN
END ENJOY ENTER ESCAPE EXAMINE EXIST EXPECT EXPLAIN EXPLORE FAIL FEAR FILL FINISH FIX FOLLOW FORM GAIN GUESS HELP HOPE
HUNT IMPROVE INCLUDE INCREASE JOIN JUMP KILL LAUGH LEARN LIKE LISTEN LIVE LOOK LOVE MANAGE MARK MATCH MISS MOVE NEED
NOTICE OFFER OPEN ORDER PAINT PASS PICK PLACE PLAN PLANT PLAY POINT PRACTICE PREPARE PRESS PULL PUSH RAIN REACH RECORD
REMAIN REMEMBER REPAIR REPEAT REPORT RETURN SAVE SEARCH SEEM SHARE SHOW SMILE SOUND START STAY STOP STUDY SUGGEST TALK
THANK TOUCH TRAVEL TRY TURN USE VISIT WAIT WALK WANT WARN WASH WATCH WISH WORK
""".split()

_ADJECTIVES = """
ABLE ACID ALIVE ALONE BASIC BLACK BLUE BRAVE BRIGHT BROWN CALM CLEAN CLEAR CLOSE COLD COOL DARK DEEP EARLY EASY EVEN
FAIR FAST FINE FIRM FLAT FREE FRESH FULL GOOD GRAND GREAT GREEN HAPPY HARD HIGH HONEST HOT HUGE KIND LARGE LATE LIGHT
LITTLE LONG LOW NEAR NEW NICE OPEN PLAIN POOR QUICK QUIET READY REAL RED RICH RIGHT ROUND SAD SAFE SHARP SHORT SIMPLE
SLOW SMALL SMART SOFT SOLID STRONG SWEET TALL THICK THIN TRUE WARM WEAK WHITE WIDE WILD WISE WRONG YOUNG
""".split()

_PREFIXABLE_VERBS = """
ACT ADD ASK BAKE BUILD CALL CARE CHECK CLEAN CLEAR CLOSE COLOR COUNT COVER CREATE DO DRAW ENTER FORM GAIN GROUP HELP JOIN
LEARN LIKE LINE LOAD LOCK MAKE MARK MATCH MOVE OPEN PACK PAINT PLACE PLAY PRINT READ RECORD SHAPE START STATE TEST TURN
USE VIEW WASH WORK WRITE
""".split()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    THEME_CSS, PITCH_SPIN_AXIS_IDEAL, VELO_BENCHMARKS,
    IVB_BENCHMARKS, HB_BENCHMARKS, GROUP_TO_DISPLAY,
)

# ---- Wrist mechanics data ----
WRIST_COMPAT = {
    'Pronator': {
        'Fastball':  ('Natural',   '#3dcc7c', "Backspin via pronation — your most natural pitch."),
        'Sinker':    ('Natural',   '#3dcc7c', "Arm-side sink via pronation — ideal motion."),
        'Changeup':  ('Natural',   '#3dcc7c', "Same arm path as fastball — natural arm-side fade."),
        'Cutter':    ('Workable',  '#f0c040', "Mild supination required — monitor elbow load."),
        'Slider':    ('Workable',  '#f0c040', "Moderate supination needed — may sacrifice sharpness."),
        'Sweeper':   ('Workable',  '#f0c040', "Less supination than curveball — achievable for most pronators."),
        'Curveball': ('Unnatural', '#f05050', "Significant supination against your natural motion — elevated elbow stress."),
    },
    'Supinator': {
        'Fastball':  ('Workable',  '#f0c040', "Achievable backspin, but arm-side ride may be reduced."),
        'Sinker':    ('Difficult', '#f05050', "Pronation-dependent — supinators typically cannot generate natural sink."),
        'Changeup':  ('Difficult', '#f05050', "Pronation required for arm-side fade — fights your natural wrist action."),
        'Cutter':    ('Natural',   '#3dcc7c', "Glove-side tilt via supination — a natural fit."),
        'Slider':    ('Natural',   '#3dcc7c', "Supination drives glove-side break — elite natural fit."),
        'Sweeper':   ('Natural',   '#3dcc7c', "Horizontal supination — sweeper is ideal for supinators."),
        'Curveball': ('Natural',   '#3dcc7c', "Full supination — curveball is the most natural pitch for supinators."),
    },
    'Neutral / Both': {
        'Fastball':  ('Flexible',  '#3a9dff', "Full wrist range — can shape backspin or add natural cut."),
        'Sinker':    ('Flexible',  '#3a9dff', "Pronation available — arm-side sink fully achievable."),
        'Changeup':  ('Flexible',  '#3a9dff', "Can pronate — natural deception pitch with full arm action."),
        'Cutter':    ('Flexible',  '#3a9dff', "Either direction available — slot determines ideal shape."),
        'Slider':    ('Flexible',  '#3a9dff', "Can supinate — glove-side break fully accessible."),
        'Sweeper':   ('Flexible',  '#3a9dff', "Full range — can generate horizontal break from either direction."),
        'Curveball': ('Flexible',  '#3a9dff', "Supination available — can execute full 12-6 rotation."),
    },
}

_CANONICAL_PAIR = {
    'Fastball':  ['Changeup', 'Slider'],
    'Sinker':    ['Changeup', 'Sweeper'],
    'Changeup':  ['Fastball', 'Slider'],
    'Slider':    ['Fastball', 'Changeup'],
    'Sweeper':   ['Fastball', 'Changeup'],
    'Curveball': ['Fastball', 'Cutter'],
    'Cutter':    ['Fastball', 'Changeup'],
}

_COMPLEMENT_WHY = {
    ('Fastball',  'Changeup'):  "Same arm action deceives hitters — velocity differential is the weapon. Arm-side fade for extra deception.",
    ('Fastball',  'Slider'):    "Glove-side break contrasts arm-side fastball life. Tunnels under your fastball for late separation.",
    ('Sinker',    'Changeup'):  "Same arm path with less velocity — tunnels your sinker while adding arm-side fade.",
    ('Sinker',    'Sweeper'):   "Horizontal glove-side break contrasts arm-side sinker. Two-way movement threat across the plate.",
    ('Changeup',  'Fastball'):  "Velocity contrast anchors your changeup. Hitters' timing collapses when speeds differ 10–15 mph.",
    ('Changeup',  'Slider'):    "Glove-side break alongside arm-side changeup. FB + CH + SL covers all movement quadrants.",
    ('Slider',    'Fastball'):  "Velocity contrast and vertical separation from the glove-side slider. FB sets up low slider exit velocity.",
    ('Slider',    'Changeup'):  "Arm-side fade alongside glove-side slider — two-way horizontal spread across the plate.",
    ('Sweeper',   'Fastball'):  "Velocity and vertical contrast with horizontal sweeper. Arm-side FB + horizontal sweeper is unsolvable timing.",
    ('Sweeper',   'Changeup'):  "Arm-side deception pairs with horizontal sweeper. FB + CH + SW covers all three movement profiles.",
    ('Curveball', 'Fastball'):  "Classic vertical contrast — curveball sets up elevated fastball. High-low tunnel collapses hitter vision.",
    ('Curveball', 'Cutter'):    "Glove-side cut adds a different look off the curveball. Both pitches play off your vertical breaking ball.",
    ('Cutter',    'Fastball'):  "Velocity and arm-side contrast with glove-side cutter. FB/cutter is one of the most effective pairings in MLB.",
    ('Cutter',    'Changeup'):  "Arm-side fade alongside glove-side cutter — covers both sides of the plate with natural velocity separation.",
}

def _get_complements(pitch_group):
    return _CANONICAL_PAIR.get(pitch_group, [])


# ──────────────────────────────────────────────────────────────────────────────
# PITCHER TYPE GUIDE — comprehensive data per wrist bias
# ──────────────────────────────────────────────────────────────────────────────
PITCHER_TYPE_GUIDE = {
    'Pronator': {
        'color': '#3dcc7c',
        'tagline': 'Arm-side power & deception master',
        'description': (
            "Pronators naturally rotate the forearm inward through the release point. "
            "This produces elite backspin on fastballs (maximum ride), arm-side run on "
            "sinkers, and the most deceiving changeup in baseball because arm action is "
            "nearly identical to the heater. The gyro slider — generated by extending "
            "pronation slightly past true topspin — is a natural weapon requiring minimal "
            "adjustment. Curveballs demand significant counter-movement against pronation, "
            "raising elbow stress."
        ),
        'arm_slot': "All slots; most effective overhand to three-quarter (55–75°). Sidearm pronators generate elite arm-side movement.",
        'wrist_orientation': "Forearm rotates inward (thumb-down) through the ball. Wrist leads the pronation; finger pressure stays behind and slightly under the ball.",
        'spin_efficiency': "88–96% on fastball/sinker/changeup. Gyro slider intentionally drops to 20–40%. Curveball typically 55–75% (fighting motion).",
        'velocity_impact': "Neutral-to-positive. Pronation is the natural power motion — elite arm speed and backspin ride go hand-in-hand.",
        'strengths': [
            "Elite fastball ride (13–18+ inches IVB) from natural backspin",
            "Best changeup platform in baseball — identical arm action to fastball",
            "Arm-side sinker achievable without mechanical change",
            "Low-maintenance gyro slider via slight pronation past center",
            "Consistent arm health — pronation is the natural throwing motion",
        ],
        'weaknesses': [
            "Hard curveball requires fighting natural motion — elevated elbow stress",
            "Limited natural glove-side break (sweeper/slider require deliberate counter-pronation)",
            "Fastball may lack horizontal ride vs. pure seam-shifted profiles",
        ],
        'arsenal': [
            {
                'name': '4-Seam Fastball', 'pct': '40–50%',
                'velo': '90–97+ mph', 'spin': '2200–2500 rpm', 'active': '90–96%',
                'hb': '+5 to +10 in', 'ivb': '+13 to +18 in',
                'why': "Natural backspin via pronation generates elite vertical ride. Primary strikeout pitch up in the zone.",
                'grip': "Standard 4-seam across the horseshoe. Pronate through the ball at release — thumb finishes down. Think 'turn a doorknob outward.'",
                'tunnels': "Changeup (same shape, 10–15 mph slower), Sinker (lower release height)",
            },
            {
                'name': 'Sinker / 2-Seam FB', 'pct': '35–45%',
                'velo': '88–95 mph', 'spin': '1900–2200 rpm', 'active': '85–92%',
                'hb': '+10 to +16 in', 'ivb': '+4 to +9 in',
                'why': "Arm-side sink from pronation is the most natural movement profile for this type. Ground ball machine.",
                'grip': "2-seam grip (index/middle on narrow seams). Pronate earlier — feel the ball 'roll' off the index finger. Sink comes from tilted spin axis.",
                'tunnels': "Changeup (arm-side fade extension), Sweeper (opposite horizontal direction)",
            },
            {
                'name': 'Changeup', 'pct': '18–28%',
                'velo': '82–88 mph', 'spin': '1600–1900 rpm', 'active': '85–95%',
                'hb': '+8 to +14 in', 'ivb': '+5 to +10 in',
                'why': "Identical arm action to fastball — pronation naturally generates arm-side fade. Elite deception pitch. Most devastating off-speed for this type.",
                'grip': "Circle-change or palmball. Deep in the hand. Pronate through at same arm speed — velocity bleeds naturally from grip, not effort reduction.",
                'tunnels': "4-Seam FB (velocity contrast), Sinker (movement contrast — change fades, sinker sinks)",
            },
            {
                'name': 'Gyro Slider', 'pct': '18–25%',
                'velo': '83–88 mph', 'spin': '2300–2600 rpm', 'active': '20–40%',
                'hb': '+2 to +7 in', 'ivb': '0 to +5 in',
                'why': "Bullet spin from extending pronation slightly past center. Late, tight, vertical-ish bite that plays off fastball. No elbow stress.",
                'grip': "Index/middle off-center (toward middle finger). Slight pronation at release creates gyro axis. Aim for 35% spin efficiency.",
                'tunnels': "4-Seam FB (same velocity band, movement separates late), Changeup (tunnel early, opposite late action)",
            },
            {
                'name': 'Cutter', 'pct': '12–20%',
                'velo': '87–92 mph', 'spin': '2200–2500 rpm', 'active': '70–82%',
                'hb': '−4 to −8 in', 'ivb': '+8 to +13 in',
                'why': "Mild counter-pronation workable for this type. Provides a glove-side option to prevent hitters from sitting arm-side.",
                'grip': "4-seam shifted toward middle finger. Slight lateral pressure at release — think 'karate chop' the ball. Limit to 15–20% to protect elbow.",
                'tunnels': "4-Seam FB (late glove-side separation), Changeup (opposite fade direction)",
            },
        ],
        'dev_tips': [
            "**Fastball ride drill**: Throw 5 fastballs focusing on 'turning a doorknob outward' at release. Film from the side — elbow should stay above the wrist through the pronation.",
            "**Changeup arm speed**: On a J-band or net, throw changeups at maximum arm speed. If you decelerate, it tips the pitch. The grip does the work.",
            "**Gyro slider feel**: Hold a ball with middle finger dominant and throw a 'spike curveball' straight into the ground. That bullet rotation is your target.",
            "**Movement plot ID**: Pronator signature = fastball cluster in upper-right quadrant (ride + arm-side run). Changeup at upper-center or upper-right, slightly below the fastball.",
            "**Video analysis cue**: Watch the forearm at ball separation. If the thumb finishes facing upward or toward the batter, you're pronating correctly. Thumb facing the sky indicates supination fighting the motion.",
            "**Elbow health**: For curveballs, use a 'Frisbee' cue (spin across the top) rather than yanking down. Limits elbow torque from forced supination.",
        ],
        'mlb_examples': [
            {
                'name': 'Gerrit Cole', 'team': 'NYY (RHP)',
                'note': "Elite pronator fastball — 18+ IVB at 96–100 mph. Gyro slider (35% eff.) and changeup (arm-side fade) are natural complements. Movement plot shows tight upper-right fastball cluster.",
                'arsenal': '4-Seam (50%) · Gyro Slider (28%) · Changeup (12%) · Curveball (10%)',
            },
            {
                'name': 'Félix Hernández', 'team': 'SEA (RHP, retired)',
                'note': "The changeup GOAT. Natural pronation drove 14-inch arm-side fade at 85 mph — identical arm path to 92 mph fastball. Four-pitch mix all leveraged the pronator profile.",
                'arsenal': 'Changeup (32%) · Sinker (28%) · 4-Seam (25%) · Curveball (15%)',
            },
            {
                'name': 'Clayton Kershaw', 'team': 'LAD (LHP)',
                'note': "Pronator LHP whose slider plays as a hard gyro offering (low spin efficiency). Fastball ride from natural pronation sets up the 12-6 curveball, which he achieves via deliberate mechanics.",
                'arsenal': '4-Seam (40%) · Slider (30%) · Curveball (20%) · Changeup (10%)',
            },
        ],
        'customization': {
            'overhand': "65°+ slot: Maximize vertical IVB on all pitches. Fastball targets top of zone. Curveball becomes more mechanically accessible. Prioritize 4-Seam + Changeup + Curveball.",
            'three_quarter': "35–65° slot: Balanced horizontal and vertical. Sinker gains significant value. Sweeper is workable as a counter-pronation pitch. Best slot for a 4-pitch mix.",
            'low_slot': "Under 35°: Arm-side horizontal movement becomes dominant. Sinker > 4-Seam as primary. Gyro slider from sidearm slot is devastating — comes from an unusual angle.",
            'power': "95+ mph velo: Lead with 4-Seam + Gyro Slider. Power profile — movement matters less. Attack up in the zone with ride.",
            'finesse': "Under 90 mph: Sinker/Changeup dominance. Spin quality and movement shape matter far more than pure velocity. Target elite IVB and HB differentials.",
            'lhp': "LHP pronators see arm-side movement run toward RHH — classic 'backdoor' threat. Changeup fades away from LHH. Gyro slider sweeps arm-side to RHH (unique look).",
        },
    },

    'Supinator': {
        'color': '#f0c040',
        'tagline': 'Glove-side wipeout artist',
        'description': (
            "Supinators naturally rotate the forearm outward through the release point, "
            "driving elite glove-side break on breaking balls. The sweeper, curveball, "
            "and slider are all natural fits — supination is the mechanical source of "
            "their movement. Fastball ride may be slightly reduced, but the high K-rate "
            "potential from wipeout breaking stuff more than compensates. Changeup and "
            "sinker require deliberate arm-path adjustment and should be used carefully."
        ),
        'arm_slot': "Most effective overhand (curveball) to three-quarter (sweeper/slider). Low-slot supinators generate unusual two-plane breaking balls.",
        'wrist_orientation': "Forearm rotates outward (thumb-up) through the ball at release. Wrist leads the supination; finger pressure stays on the side or top of the ball depending on the pitch.",
        'spin_efficiency': "75–95% on breaking balls. Fastball typically 82–90% (slight efficiency loss vs. pronators). Curveball/sweeper both benefit from full supination.",
        'velocity_impact': "Slightly reduced fastball ride vs. pronators; exceptional breaking ball movement compensates. Breaking ball velocity sits 3–7 mph below fastball.",
        'strengths': [
            "Elite glove-side break — sweeper, curveball, and slider are all natural",
            "Wipeout potential on breaking balls from full supination",
            "Cutter is a natural complement — glove-side cut with similar motion",
            "High strikeout ceiling with elite breaking ball pairing",
            "Curveball depth from full supination is among the best in baseball",
        ],
        'weaknesses': [
            "Changeup requires significant arm-path change — tips the pitch or risks injury",
            "Sinker is difficult — pronation-dependent movement fights natural motion",
            "Fastball arm-side run may be reduced; can be predictable without mixing",
            "Overuse of breaking balls can elevate elbow and forearm stress",
        ],
        'arsenal': [
            {
                'name': '4-Seam Fastball', 'pct': '30–42%',
                'velo': '91–97+ mph', 'spin': '2100–2400 rpm', 'active': '82–92%',
                'hb': '+2 to +8 in', 'ivb': '+10 to +16 in',
                'why': "Achievable with deliberate backspin focus. Reduced arm-side run but still generates ride. Primary pitch to set up glove-side breaking balls.",
                'grip': "Standard 4-seam. Focus on staying behind the ball at release — resist the natural supination by keeping the fingers directly behind at 12 o'clock.",
                'tunnels': "Sweeper/Slider (same tunnel, opposite horizontal break), Curveball (same start point, more vertical drop)",
            },
            {
                'name': 'Sweeper', 'pct': '25–35%',
                'velo': '80–86 mph', 'spin': '2400–2800 rpm', 'active': '75–90%',
                'hb': '−15 to −22 in', 'ivb': '0 to +5 in',
                'why': "The sweeper is the signature pitch for supinators. Full forearm supination drives maximum horizontal sweep. The most natural offering for this type.",
                'grip': "Slider grip shifted toward the thumb side. Supinate aggressively at release — pull down a window shade. Maximize horizontal finger drag across the ball.",
                'tunnels': "4-Seam FB (same arm path, horizontal separation late), Changeup (velocity differential if available)",
            },
            {
                'name': 'Curveball', 'pct': '18–26%',
                'velo': '75–82 mph', 'spin': '2600–3200 rpm', 'active': '80–93%',
                'hb': '−6 to −12 in', 'ivb': '−10 to −16 in',
                'why': "Full supination produces elite 11-to-5 or 12-to-6 curveball action. The most natural breaking pitch for this type — deep tilt from the wrist.",
                'grip': "Index-middle over the top seam. Drive the elbow ahead of the hand ('lead with the elbow') and let supination pull the ball downward. 12-to-6 from overhand slot.",
                'tunnels': "4-Seam FB (high-low tunnel — same height at tunnel point, splits vertically), Cutter (lateral vs. vertical contrast)",
            },
            {
                'name': 'Cutter', 'pct': '15–22%',
                'velo': '87–92 mph', 'spin': '2200–2500 rpm', 'active': '70–83%',
                'hb': '−4 to −9 in', 'ivb': '+8 to +13 in',
                'why': "Natural supination fit — glove-side cut at fastball velocity. Hard to barrel. Works off the fastball with same arm speed, different finish.",
                'grip': "4-seam shifted toward index finger. Slight supination at release — 'shave' across the outside of the ball. Cut should be 4–8 inches glove-side.",
                'tunnels': "4-Seam FB (velocity match, late movement diverges), Curveball (similar glove-side start, cutter stays up)",
            },
            {
                'name': 'Changeup (Optional)', 'pct': '10–16%',
                'velo': '82–88 mph', 'spin': '1600–1900 rpm', 'active': '80–92%',
                'hb': '+6 to +12 in', 'ivb': '+4 to +9 in',
                'why': "Requires deliberate arm-path modification. If mastered, provides the one arm-side offering to prevent hitters from sitting glove-side. High upside but not natural.",
                'grip': "Deep circle-change. Commit to pronation cue ('turn the doorknob') against your natural supination. Limit usage until mechanics are automatic.",
                'tunnels': "4-Seam FB (velocity drop), Sweeper (opposite horizontal direction — creates both-ways threat)",
            },
        ],
        'dev_tips': [
            "**Sweeper feel drill**: Hold a ball and try to 'wipe a whiteboard' horizontally at release. The forearm should rotate outward maximally. Film from behind the mound to confirm horizontal finger drag.",
            "**Curveball lead-elbow drill**: Throw curveballs with an exaggerated 'elbow first' movement. The elbow should cross the midline before the hand fires. This prevents yanking down and protects the UCL.",
            "**Movement plot ID**: Supinator signature = breaking ball cluster in the left quadrant (glove-side) with fastball in the upper center. Sweeper and curveball should form a clear glove-side spread.",
            "**Video analysis cue**: At ball separation, the thumb should be pointing upward or toward the sky. If the thumb rotates inward (pronation attempt on a breaking ball), the pitch loses its shape.",
            "**Changeup development**: If adding a changeup, use a heavy towel drill — simulate full arm speed while gripping as deep as possible. Accept that this pitch may take 2–3 seasons to automate.",
            "**Arm care**: High supination load on breaking balls increases flexor/UCL demand. Prioritize forearm flexibility, J-band stretching, and limiting breaking ball volume to 40% or less in practice.",
        ],
        'mlb_examples': [
            {
                'name': 'Spencer Strider', 'team': 'ATL (RHP)',
                'note': "Sweeper-dominant supinator who averaged 18+ inches of glove-side break. 4-Seam + Sweeper binary was nearly unsolvable in 2023 (228 K/162 IP). Pure elite supinator signature.",
                'arsenal': '4-Seam (55%) · Sweeper (38%) · Slider (7%)',
            },
            {
                'name': 'Sandy Alcantara', 'team': 'MIA (RHP)',
                'note': "Supinator with elite curveball-cutter-sweeper combination. Four pitches all leverage glove-side tilt. Sinker usage unusual for the type — he compensates with elite extension (7.0 ft).",
                'arsenal': 'Sinker (28%) · Changeup (22%) · Slider (20%) · Cutter (18%) · 4-Seam (12%)',
            },
            {
                'name': 'Freddie Peralta', 'team': 'MIL (RHP)',
                'note': "Classic supinator — curveball and slider account for 60%+ of his arsenal. High spin rates on all breaking pitches. Fastball ride reduced but plays off the breaking ball tunnels.",
                'arsenal': '4-Seam (35%) · Curveball (30%) · Slider (25%) · Changeup (10%)',
            },
        ],
        'customization': {
            'overhand': "65°+ slot: Curveball becomes your best pitch — full 12-to-6 or 11-to-5 drop. Pair with 4-Seam up in the zone. Prioritize 4-Seam + Curveball + Cutter.",
            'three_quarter': "35–65° slot: Sweeper maximizes horizontal break from this plane. Best slot for a sweeper-dominated profile. Curveball takes a 10-to-4 shape.",
            'low_slot': "Under 35°: Sweeper from sidearm slot becomes a 'frisbee' with 25+ inches of break. Extremely rare and devastating. Fastball plays like a rising backdoor pitch.",
            'power': "95+ mph velo: 4-Seam + Sweeper binary is elite — nearly unsolvable. Attack with velo-movement combo. Limit curveball in favor of the sweeper.",
            'finesse': "Under 90 mph: Curveball depth and cutter sharpness become primary weapons. Spin efficiency must be 85%+ on breaking balls. Work sequencing and tunneling over raw stuff.",
            'lhp': "LHP supinators' sweeper runs away from RHH (glove-side from pitcher perspective = arm-side for RHH). Curveball buckles LHH knees. Elite platoon splitter — feast on same-handed hitters.",
        },
    },

    'Neutral / Both': {
        'color': '#3a9dff',
        'tagline': 'Versatile arsenal architect',
        'description': (
            "Neutral pitchers have full wrist range — they can deliberately produce both "
            "pronation-based and supination-based movement through grip adjustments and "
            "deliberate release point changes. This makes them the most versatile pitcher "
            "type, capable of developing any pitch with ceiling. The tradeoff is that "
            "without a strong wrist bias, extreme movement (elite IVB or elite sweep) "
            "requires more intentional mechanical execution. The best neutral pitchers "
            "use 4–5 pitch mixes with distinct movement profiles in every direction."
        ),
        'arm_slot': "Effective at all arm slots. Deliberate adjustments unlock pitch types from any angle. Three-quarter is the most common slot for this profile.",
        'wrist_orientation': "Full range — can pronate for arm-side movement or supinate for glove-side break without significant mechanical strain.",
        'spin_efficiency': "Variable by pitch type — 88–95% on fastball, 75–90% on breaking balls. Highest ceiling for spin efficiency diversity across an arsenal.",
        'velocity_impact': "Neutral. Can achieve elite characteristics across all pitch types with proper mechanical cues.",
        'strengths': [
            "Maximum arsenal depth — can develop any pitch type to elite level",
            "Both-ways horizontal threat — arm-side and glove-side movement available",
            "Highest adaptability to arm slot changes or injury-induced adjustments",
            "Tunneling advantage — can pair pitches that move in opposite directions",
            "Difficult to scout — no obvious movement bias that hitters can exploit",
        ],
        'weaknesses': [
            "May lack extreme signatures of pure pronator or supinator",
            "Requires more deliberate mechanical cues to achieve elite-level break",
            "Arsenal development takes longer — must master multiple wrist positions",
            "Without a dominant pitch, can become predictable if sequencing lacks purpose",
        ],
        'arsenal': [
            {
                'name': '4-Seam Fastball', 'pct': '35–45%',
                'velo': '90–97+ mph', 'spin': '2100–2500 rpm', 'active': '88–95%',
                'hb': '+5 to +10 in', 'ivb': '+12 to +18 in',
                'why': "Foundation of the arsenal. Full wrist range allows deliberate tuning of arm-side vs. cut. Work toward maximum IVB via spin axis dialing.",
                'grip': "Standard 4-seam. Experiment with spin axis — pronation at release for more ride and run, neutral finger position for pure backspin and max IVB.",
                'tunnels': "All other pitches — fastball is the reference for every tunnel relationship",
            },
            {
                'name': 'Slider or Sweeper', 'pct': '22–30%',
                'velo': '82–87 mph', 'spin': '2400–2700 rpm', 'active': '65–85%',
                'hb': '−10 to −18 in', 'ivb': '+1 to +8 in',
                'why': "Glove-side break available via deliberate supination. With full wrist range, can shape from sharp slider to wide sweeper. K-rate pitch.",
                'grip': "Off-center grip toward index finger for slider, toward middle finger for sweeper. Supinate at release — feel the thumb rotating upward.",
                'tunnels': "4-Seam FB (horizontal separation at tunnel point), Changeup (opposite horizontal move)",
            },
            {
                'name': 'Changeup', 'pct': '18–25%',
                'velo': '82–88 mph', 'spin': '1500–1900 rpm', 'active': '80–92%',
                'hb': '+8 to +13 in', 'ivb': '+4 to +10 in',
                'why': "Full pronation range allows natural arm-side fade. Velocity separation is the weapon. Pairs perfectly with the slider/sweeper for two-way horizontal coverage.",
                'grip': "Circle-change deep in the palm. Pronate deliberately at release — same arm speed as fastball. Aim for 10–15 mph velocity separation.",
                'tunnels': "4-Seam FB (velocity separation), Slider (opposite horizontal — together they cover both sides of the plate)",
            },
            {
                'name': 'Curveball', 'pct': '12–20%',
                'velo': '76–82 mph', 'spin': '2600–3000 rpm', 'active': '78–92%',
                'hb': '−5 to −11 in', 'ivb': '−10 to −15 in',
                'why': "Full supination range makes curveball mechanically accessible. Provides elite vertical separation from the fastball. High-low tunnel is the foundation.",
                'grip': "Index-middle over the top. Lead with the elbow, supinate through. Target 11-to-5 rotation for the best shape from three-quarter slot.",
                'tunnels': "4-Seam FB (elite high-low — same tunnel height, splits vertically), Cutter (both glove-side, different depth)",
            },
            {
                'name': 'Cutter', 'pct': '12–18%',
                'velo': '87–92 mph', 'spin': '2200–2500 rpm', 'active': '70–82%',
                'hb': '−4 to −8 in', 'ivb': '+8 to +13 in',
                'why': "Available from both pronator and supinator mechanics — neutral pitchers can dial in the exact shape they want. Glove-side velocity pitch that plays off the fastball.",
                'grip': "4-seam shifted toward index finger, slight supination at release. Aim for 4–7 inches of glove-side cut at 87–92 mph.",
                'tunnels': "4-Seam FB (same velocity range, late glove-side separation), Changeup (opposite horizontal with velocity gap)",
            },
        ],
        'dev_tips': [
            "**Spin axis tuning**: Film every pitch from behind the mound. Work with a Rapsodo or Trackman to map your spin axis for each pitch. The goal is 4–5 distinct spin axes that generate clearly different movement profiles.",
            "**Deliberate pronation/supination drills**: Alternate between 'doorknob outward' (pronation) and 'window shade pull' (supination) release cues on consecutive throws. Train both motions until both are automatic.",
            "**Movement plot segmentation**: Your arsenal should show clear clusters in at least 3 quadrants of the HB vs. IVB chart. If all pitches cluster together, work on spin axis separation.",
            "**Tunneling lab**: Use video or a pitching simulator to confirm your pitches share the same 'tunnel point' through the hitting zone. They should look identical until 25–30 feet from home plate.",
            "**Pitch identity**: Assign each pitch a job (fastball = north zone, slider = glove-side chase, changeup = arm-side depth, curve = vertical separation). Neutral pitchers must be intentional about sequencing.",
            "**Identification via break plot**: Neutral signature = pitches spread across all four quadrants of the movement chart. No obvious clustering in one direction. Elite neutral pitchers form a 'cross' or 'star' pattern.",
        ],
        'mlb_examples': [
            {
                'name': 'Justin Verlander', 'team': 'HOU/NYM (RHP)',
                'note': "Textbook neutral mechanics with a five-pitch mix. 4-Seam (high ride), Slider (sharp glove-side), Curveball (deep vertical), Changeup (arm-side), Cutter (tight glove-side). All four quadrants covered.",
                'arsenal': '4-Seam (35%) · Curveball (25%) · Slider (20%) · Changeup (15%) · Cutter (5%)',
            },
            {
                'name': 'Zac Gallen', 'team': 'ARI (RHP)',
                'note': "Elite movement diversity from neutral wrist position. Four distinct pitches cover opposite movement planes. Known for high spin efficiency across all offerings — deliberate mechanics.",
                'arsenal': '4-Seam (30%) · Curveball (22%) · Changeup (22%) · Sinker (14%) · Slider (12%)',
            },
            {
                'name': 'Chris Sale', 'team': 'ATL (LHP)',
                'note': "Extreme three-quarter slot with neutral mechanics enabling both glove-side slider and arm-side changeup. Velocity-plus-movement combination from an unusual arm angle.",
                'arsenal': '4-Seam (40%) · Slider (32%) · Changeup (18%) · Curveball (10%)',
            },
        ],
        'customization': {
            'overhand': "65°+ slot: Curveball becomes elite — 12-to-6 drop available. Fastball ride maximized. Prioritize 4-Seam + Curveball + Slider three-pitch core.",
            'three_quarter': "35–65° slot: Perfect slot for the neutral type — sweeper, slider, cutter, curveball, and changeup are all accessible. Build the full 5-pitch arsenal.",
            'low_slot': "Under 35°: Horizontal movement dominates both sides. Slider/sweeper from this slot sweeps dramatically. Curveball flattens — consider replacing with a gyro-style pitch.",
            'power': "95+ mph: Power-breaking ball combination. Sweeper + 4-Seam is elite. Keep the arsenal at 3 pitches for command efficiency — don't over-develop.",
            'finesse': "Under 90 mph: Maximum arsenal depth is the strategy. 5 pitches that tunnel perfectly. Movement quality, spin efficiency, and sequencing matter far more than velocity.",
            'lhp': "LHP neutral pitchers have the most versatile platoon profiles — can attack both RHH and LHH with appropriate pitch selection. Changeup away from RHH, slider away from LHH.",
        },
    },
}

def _infer_wrist_type(pitch_group, ivb, hb, spin_eff_pct):
    """
    Infer pronator/supinator bias from pitch movement profile.
    Returns (wrist_type, confidence, note).
    """
    h, v = abs(hb), abs(ivb)

    if pitch_group == 'Sweeper':
        if h >= 16: return ('Supinator', 'High confidence',    f"{h:.0f} in horizontal sweep — strong supinator signature")
        if h >= 11: return ('Supinator', 'Moderate confidence', f"{h:.0f} in sweep — consistent with supinator arm action")
        return ('Neutral / Both', 'Low confidence', "Below-average sweep — atypical sweeper mechanics")

    if pitch_group == 'Curveball':
        if ivb <= -10 and spin_eff_pct >= 75:
            return ('Supinator', 'High confidence',    f"{v:.0f} in drop + {spin_eff_pct}% efficiency — full supination")
        if ivb <= -7:
            return ('Supinator', 'Moderate confidence', f"{v:.0f} in drop — supinator curveball signature")
        if spin_eff_pct < 50:
            return ('Neutral / Both', 'Low confidence', f"Low efficiency ({spin_eff_pct}%) — unusual curveball mechanics")
        return ('Supinator', 'Moderate confidence', "Curveball shape — typically driven by supination")

    if pitch_group == 'Fastball':
        # Tread Athletics primary signal: spin efficiency is the strongest indicator
        if spin_eff_pct >= 95:
            return ('Pronator', 'High confidence',    f"{spin_eff_pct}% spin efficiency — pure backspin, elite pronator signature (Tread ≥95% threshold)")
        if spin_eff_pct >= 88 and ivb >= 14:
            return ('Pronator', 'High confidence',    f"{spin_eff_pct}% efficiency + {ivb:.0f} in ride — strong pronator backspin")
        if spin_eff_pct >= 80:
            return ('Pronator', 'Moderate confidence', f"{spin_eff_pct}% efficiency — solid backspin, pronation likely")
        if spin_eff_pct < 70:
            return ('Supinator', 'Moderate confidence', f"{spin_eff_pct}% efficiency — reduced backspin (Tread 60-90% supinator range)")
        if h >= 13:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in arm-side run — seam-shifted supinator fastball")
        return ('Neutral / Both', 'Low confidence', "Mixed efficiency/movement — neutral or seam-shifted mechanics")

    if pitch_group == 'Sinker':
        if h >= 14 and ivb <= 5:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in run + heavy sink — seam-shifted supinator signature")
        if ivb >= 7 and h <= 8:
            return ('Pronator', 'Moderate confidence',  f"Vertical-dominant sinker ({ivb:.0f} IVB) — pronation bias")
        return ('Neutral / Both', 'Low confidence', "Standard sinker movement — mechanics ambiguous")

    if pitch_group == 'Slider':
        if spin_eff_pct <= 35 and h <= 7 and v <= 5:
            return ('Pronator', 'High confidence',    f"Gyro slider ({spin_eff_pct}% eff, {h:.0f}/{v:.0f} in break) — pronator bullet spin")
        if h >= 10 and spin_eff_pct >= 55:
            return ('Supinator', 'High confidence',    f"Active-spin slider with {h:.0f} in sweep — supinator signature")
        if h >= 7:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in glove-side break — supination indicator")
        return ('Neutral / Both', 'Low confidence', "Mixed slider mechanics")

    if pitch_group == 'Changeup':
        if h >= 10 and spin_eff_pct >= 80:
            return ('Pronator', 'High confidence',    f"{h:.0f} in arm-side fade + {spin_eff_pct}% efficiency — natural pronation")
        if h >= 7 and ivb >= 6:
            return ('Pronator', 'Moderate confidence', f"Arm-side fade changeup ({h:.0f} in HB) — pronation signature")
        if h < 5:
            return ('Supinator', 'Low confidence',     "Low arm-side fade — may indicate supinator fighting pronation on changeup")
        return ('Pronator', 'Low confidence', "Changeup arm action typically reflects pronation; add FB to arsenal for efficiency comparison")

    if pitch_group == 'Cutter':
        if h >= 8:
            return ('Supinator', 'Moderate confidence', f"{h:.0f} in glove-side cut — supination-driven")
        if h <= 4 and spin_eff_pct >= 70:
            return ('Pronator', 'Low confidence',       "Tight cutter — pronation limiting glove-side break")
        return ('Neutral / Both', 'Low confidence', "Balanced cutter — mechanics ambiguous")

    return ('Neutral / Both', 'Low confidence', "Insufficient data to determine wrist bias")


def _infer_wrist_type_arsenal(pitches):
    """
    Arsenal-wide wrist bias detection using Tread Athletics framework.
    pitches: list of dicts with keys: pitch_group, velo, spin, spin_eff, ivb, hb
    Returns (wrist_type, confidence, signals_list).
    """
    signals = []  # list of (wrist_type, weight, note)

    fb = next((p for p in pitches if p['pitch_group'] == 'Fastball'), None)
    si = next((p for p in pitches if p['pitch_group'] == 'Sinker'), None)
    fb_ref = fb or si   # primary velocity reference
    ch = next((p for p in pitches if p['pitch_group'] == 'Changeup'), None)
    sw = next((p for p in pitches if p['pitch_group'] == 'Sweeper'), None)
    sl = next((p for p in pitches if p['pitch_group'] == 'Slider'), None)
    cu = next((p for p in pitches if p['pitch_group'] == 'Curveball'), None)

    # ── Signal 1: Fastball spin efficiency (Tread primary) ──────────────────
    if fb_ref:
        eff = fb_ref['spin_eff']
        if eff >= 95:
            signals.append(('Pronator', 3, f"FB spin efficiency {eff}% ≥ 95% — pure backspin, pronator signature (Tread)"))
        elif eff >= 88:
            signals.append(('Pronator', 2, f"FB spin efficiency {eff}% (88-95%) — solid backspin, pronation likely"))
        elif eff >= 75:
            signals.append(('Neutral / Both', 1, f"FB spin efficiency {eff}% (75-88%) — mixed or neutral wrist"))
        else:
            signals.append(('Supinator', 2, f"FB spin efficiency {eff}% (<75%) — reduced backspin, supinator range (Tread 60-90%)"))

    # ── Signal 2: Changeup vs. fastball efficiency (Tread red flag) ─────────
    if fb_ref and ch:
        fb_eff = fb_ref['spin_eff']
        ch_eff = ch['spin_eff']
        if ch_eff > fb_eff + 2:
            signals.append(('Supinator', 3,
                f"Changeup efficiency ({ch_eff}%) > FB ({fb_eff}%) — Tread red flag: supinator side-spinning changeup"))
        else:
            signals.append(('Pronator', 1,
                f"Changeup efficiency ({ch_eff}%) ≤ FB ({fb_eff}%) — natural pronation signature"))

    # ── Signal 3: Velocity deficit to changeup (Tread 8-10 mph = pronator) ──
    if fb_ref and ch:
        diff = fb_ref['velo'] - ch['velo']
        if diff >= 10:
            signals.append(('Pronator', 2, f"FB→CH gap {diff:.0f} mph (≥10) — pronator-typical velocity deficit"))
        elif diff >= 7:
            signals.append(('Neutral / Both', 1, f"FB→CH gap {diff:.0f} mph (7-10) — average range"))
        else:
            signals.append(('Supinator', 1, f"FB→CH gap {diff:.0f} mph (<7) — tight; pronation may not be natural"))

    # ── Signal 4: Breaking ball velocity deficit to FB (5-8 mph = supinator)─
    bb = sw or sl
    if fb_ref and bb:
        diff_bb = fb_ref['velo'] - bb['velo']
        if 5 <= diff_bb <= 8:
            signals.append(('Supinator', 2,
                f"FB→{bb['pitch_group']} gap {diff_bb:.0f} mph (5-8) — supinator-typical breaking ball deficit"))
        elif diff_bb > 11:
            signals.append(('Pronator', 1,
                f"Wide FB→{bb['pitch_group']} gap {diff_bb:.0f} mph — pronator-like spacing"))

    # ── Signal 5: Sweeper horizontal break ───────────────────────────────────
    if sw:
        h_sw = abs(sw['hb'])
        if h_sw >= 18:
            signals.append(('Supinator', 3, f"{h_sw:.0f} in sweeper break — elite supinator signature"))
        elif h_sw >= 13:
            signals.append(('Supinator', 2, f"{h_sw:.0f} in horizontal sweep — supinator indicator"))

    # ── Signal 6: Gyro slider (pronator bullet spin) ─────────────────────────
    if sl and sl['spin_eff'] <= 35:
        signals.append(('Pronator', 3,
            f"Gyro slider {sl['spin_eff']}% efficiency — pronator bullet spin (Tread pronator triangle)"))

    # ── Signal 7: Curveball ──────────────────────────────────────────────────
    if cu and cu['ivb'] <= -10 and cu['spin_eff'] >= 75:
        signals.append(('Supinator', 2, f"Curveball {cu['ivb']:.0f} IVB + {cu['spin_eff']}% eff — full supination"))

    # ── Fall back to single-pitch for single-entry arsenals ─────────────────
    if not signals:
        if pitches:
            p = pitches[0]
            wt, wc, wn = _infer_wrist_type(p['pitch_group'], p['ivb'], p['hb'], p['spin_eff'])
            return wt, wc, [wn]
        return 'Neutral / Both', 'Low confidence', ['Insufficient data']

    # ── Weighted vote tally ──────────────────────────────────────────────────
    votes = {'Pronator': 0, 'Supinator': 0, 'Neutral / Both': 0}
    for wt, weight, _ in signals:
        votes[wt] += weight

    winner = max(votes, key=votes.get)
    total = sum(votes.values())
    winner_pct = votes[winner] / total if total else 0

    confidence = ('High confidence' if winner_pct >= 0.70 else
                  'Moderate confidence' if winner_pct >= 0.50 else
                  'Low confidence')
    notes = [n for wt, w, n in signals if wt == winner] or [n for _, _, n in signals[:2]]
    return winner, confidence, notes


# ── MLB Comps by Arm Slot × Wrist Bias ─────────────────────────────────────
MLB_COMPS_BY_SLOT = {
    'Pronator': {
        'Overhand':      'Gerrit Cole (NYY) · Max Fried (NYY) · Pablo López (MIN)',
        'High 3/4':      'Félix Hernández (ret.) · Kevin Gausman (TOR) · Logan Webb (SF)',
        'Three-Quarter': 'Clayton Kershaw (LAD) · Dylan Cease (SD) · Charlie Morton (MIL)',
        'Low 3/4':       'Joe Musgrove (SD) · Marco Gonzales (SEA) · Jake Odorizzi (ret.)',
        'Sidearm':       'Chris Devenski (ret.) · Pat Neshek (ret.)',
    },
    'Supinator': {
        'Overhand':      'Spencer Strider (ATL) · Freddie Peralta (MIL) · Walker Buehler (LAD)',
        'High 3/4':      'Sandy Alcantara (MIA) · Josh Hader (HOU) · Shane Bieber (CLE)',
        'Three-Quarter': 'Blake Snell (SF) · Corbin Burnes (BAL) · Robbie Ray (SF)',
        'Low 3/4':       'Chris Sale (ATL) · Andrew Miller (ret.) · Randy Johnson (ret., LHP)',
        'Sidearm':       'Tyler Rogers (SF) · Brad Ziegler (ret.)',
    },
    'Neutral / Both': {
        'Overhand':      'Zack Wheeler (PHI) · Aaron Nola (PHI) · Marcus Stroman (CHC)',
        'High 3/4':      'Zac Gallen (ARI) · Logan Gilbert (SEA) · Sonny Gray (STL)',
        'Three-Quarter': 'Carlos Rodón (NYY) · Dan Haren (ret.) · Lance Lynn (ret.)',
        'Low 3/4':       'José Quintana (ret.) · Rich Hill (ret.)',
        'Sidearm':       "Darren O'Day (ret.) · Mike Myers (ret., LHP)",
    },
}

# Pitch type colors for movement chart
PT_COLOR = {
    'Fastball':  '#f05050',
    'Sinker':    '#f0a050',
    'Cutter':    '#f0c040',
    'Slider':    '#3dcc7c',
    'Sweeper':   '#20aa60',
    'Curveball': '#3a9dff',
    'Changeup':  '#b47fff',
}

st.set_page_config(page_title="Pitch Shape Advisor · AlpAnalytics", layout="wide", page_icon="🔬", initial_sidebar_state="expanded")
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-logo">⚾ AlpAnalytics</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="info-box" style="margin-top:1rem;">
    All recommendations are benchmarked against MLB averages and elite thresholds only.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <p class="app-title">Pitch Shape Advisor</p>
    <p class="app-subtitle">AlpAnalytics · MLB-Calibrated · Shape & Spin Targets</p>
</div>
""", unsafe_allow_html=True)

# ---- Session state ----
if 'arsenal' not in st.session_state:
    st.session_state.arsenal = []

# ---- Shared Pitcher Settings ----
st.markdown('<div class="section-header">Pitcher Profile</div>', unsafe_allow_html=True)
_hand_col, _slot_col = st.columns([1, 3])
with _hand_col:
    handedness = st.radio("Handedness", ["R", "L"], horizontal=True,
                          format_func=lambda x: "RHP" if x == "R" else "LHP")
with _slot_col:
    _sc1, _sc2, _sc3 = st.columns(3)
    with _sc1:
        rel_height = st.slider("Release Height (ft)", min_value=4.0, max_value=8.0, value=6.2, step=0.05)
    with _sc2:
        rel_side = st.slider("Release Side (ft)", min_value=-4.0, max_value=4.0, value=1.5, step=0.05,
                             help="Distance from center of rubber. Positive = arm side.")
    with _sc3:
        ext = st.slider("Extension (ft)", min_value=4.0, max_value=7.5, value=6.2, step=0.05)

arm_angle = int(np.clip(np.degrees(np.arctan2(max(0.0, rel_height - 5.0), max(0.01, abs(rel_side)))), 0, 90))
_slot_name = ('Sidearm' if arm_angle < 15 else 'Low 3/4' if arm_angle < 30 else
              'Three-Quarter' if arm_angle < 50 else 'High 3/4' if arm_angle < 65 else 'Over the Top')
st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2a3348;border-radius:4px;padding:0.5rem 0.9rem;
            margin-top:0.4rem;display:flex;align-items:center;gap:0.8rem;">
    <span style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">Calculated Arm Angle</span>
    <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;font-weight:700;color:#f0c040;">{arm_angle}°</span>
    <span style="color:#505a70;font-size:0.75rem;">{_slot_name}</span>
</div>
""", unsafe_allow_html=True)

# ---- Arsenal Builder ----
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Arsenal Builder</div>', unsafe_allow_html=True)
st.markdown("<span style='color:#8a94aa;font-size:0.82rem;'>Enter each pitch separately and click Add. Analyze after all pitches are added.</span>", unsafe_allow_html=True)
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

_pt_col, _velo_col = st.columns([2, 1])
with _pt_col:
    pitch_group = st.selectbox("Pitch Type", list(GROUP_TO_DISPLAY.keys()),
                               format_func=lambda k: GROUP_TO_DISPLAY[k])
with _velo_col:
    velo = st.slider("Velocity (mph)", min_value=60.0, max_value=105.0, value=93.0, step=0.5)

c_left, c_right = st.columns(2)
with c_left:
    st.markdown("**Spin**")
    spin_rate = st.slider("Spin Rate (rpm)", min_value=1200, max_value=3800, value=2300, step=10)
    spin_eff  = st.slider("Spin Efficiency (%)", min_value=0, max_value=110, value=92, step=1,
                          help="Transverse spin percentage. 100% = all spin generates movement.")
    spin_axis = st.slider("Spin Axis (°)", min_value=0, max_value=359, value=190, step=1,
                          help="Clock face: 0/360=6 o'clock, 90=9 o'clock, 180=12 o'clock, 270=3 o'clock")
with c_right:
    st.markdown("**Movement**")
    ivb = st.slider("Induced Vert Break (in)", min_value=-25.0, max_value=25.0, value=14.0, step=0.5,
                    help="Positive = ride (resists gravity), negative = depth/drop.")
    hb  = st.slider("Horizontal Break (in)", min_value=-25.0, max_value=25.0, value=8.0, step=0.5,
                    help="Arm-side adjusted. Positive = arm-side.")

# Single-pitch wrist preview
wrist_type_cur, wrist_conf_cur, wrist_note_cur = _infer_wrist_type(pitch_group, ivb, hb, spin_eff)
_wc_cur = '#3dcc7c' if wrist_type_cur == 'Pronator' else '#f0c040' if wrist_type_cur == 'Supinator' else '#3a9dff'
st.markdown(f"""
<div style="background:#1c2230;border:1px solid #2a3348;border-radius:4px;padding:0.5rem 0.9rem;
            margin-top:0.5rem;display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;">
    <span style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;">Single-Pitch Wrist Indicator</span>
    <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;color:{_wc_cur};">{wrist_type_cur}</span>
    <span style="color:{_wc_cur};font-size:0.72rem;font-weight:600;opacity:0.75;">{wrist_conf_cur}</span>
    <span style="color:#505a70;font-size:0.75rem;flex:1;">{wrist_note_cur}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
_btn1, _btn2, _btn3 = st.columns([2, 1.5, 2])
with _btn1:
    _add_clicked = st.button("➕ Add Pitch to Arsenal", use_container_width=True)
with _btn2:
    _clear_clicked = st.button("🗑️ Clear Arsenal", use_container_width=True)
with _btn3:
    analyze = st.button("🔬 Analyze Arsenal", use_container_width=True,
                        disabled=(len(st.session_state.arsenal) == 0))

if _add_clicked:
    st.session_state.arsenal.append({
        'pitch_group': pitch_group, 'velo': velo, 'spin': spin_rate,
        'spin_eff': spin_eff, 'spin_axis': spin_axis, 'ivb': ivb, 'hb': hb,
    })
    st.rerun()

if _clear_clicked:
    st.session_state.arsenal = []
    st.rerun()

# ---- Arsenal Preview Table ----
if st.session_state.arsenal:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Current Arsenal</div>', unsafe_allow_html=True)
    _ars_preview = ""
    for _pi, _p in enumerate(st.session_state.arsenal):
        _pcolor = PT_COLOR.get(_p['pitch_group'], '#e8eaf0')
        _ars_preview += f"""
        <tr style="border-bottom:1px solid #1c2230;">
          <td style="padding:7px 14px;color:{_pcolor};font-weight:700;
                     font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;">#{_pi+1} {GROUP_TO_DISPLAY[_p['pitch_group']]}</td>
          <td style="padding:7px 12px;color:#e8eaf0;">{_p['velo']:.1f} mph</td>
          <td style="padding:7px 12px;color:#e8eaf0;">{_p['spin']:,} rpm</td>
          <td style="padding:7px 12px;color:#e8eaf0;">{_p['spin_eff']}%</td>
          <td style="padding:7px 12px;color:#e8eaf0;">{_p['ivb']:+.1f} in</td>
          <td style="padding:7px 12px;color:#e8eaf0;">{_p['hb']:+.1f} in</td>
        </tr>"""
    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;background:#141820;border:1px solid #2a3348;border-radius:6px;">
      <thead><tr style="background:#1c2230;">
        <th style="padding:7px 14px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Pitch</th>
        <th style="padding:7px 12px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Velo</th>
        <th style="padding:7px 12px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Spin</th>
        <th style="padding:7px 12px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Active%</th>
        <th style="padding:7px 12px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">IVB</th>
        <th style="padding:7px 12px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">HB</th>
      </tr></thead>
      <tbody>{_ars_preview}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
    _rm_cols = st.columns(min(len(st.session_state.arsenal), 6))
    for _pi in range(len(st.session_state.arsenal)):
        with _rm_cols[_pi % 6]:
            if st.button(f"Remove #{_pi+1}", key=f"rm_{_pi}"):
                st.session_state.arsenal.pop(_pi)
                st.rerun()
else:
    st.info("No pitches added yet. Enter a pitch above and click ➕ Add Pitch to Arsenal.")

if analyze and st.session_state.arsenal:
    pitches = st.session_state.arsenal
    ars_wrist, ars_conf, ars_signals = _infer_wrist_type_arsenal(pitches)
    _wc_ars = '#3dcc7c' if ars_wrist == 'Pronator' else '#f0c040' if ars_wrist == 'Supinator' else '#3a9dff'

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Arsenal Analysis</div>', unsafe_allow_html=True)

    # ── Arsenal-wide wrist bias banner ──────────────────────────────────────
    _sig_html = "".join(
        f"<div style='color:#8a94aa;font-size:0.78rem;margin-top:3px;'>▸ {s}</div>"
        for s in ars_signals
    )
    st.markdown(f"""
    <div style="background:#1c2230;border:1px solid #2a3348;border-left:4px solid {_wc_ars};
                border-radius:6px;padding:1rem 1.4rem;margin-bottom:1rem;">
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.4rem;">
        <div>
          <span style="color:#8a94aa;font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
                       text-transform:uppercase;margin-right:0.7rem;">Arsenal Wrist Bias (Tread Framework)</span>
          <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:800;
                       color:{_wc_ars};">{ars_wrist}</span>
          <span style="color:{_wc_ars};font-size:0.75rem;font-weight:600;margin-left:0.6rem;
                       opacity:0.8;">{ars_conf}</span>
        </div>
      </div>
      {_sig_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Movement overlay + Velocity differential ────────────────────────────
    _mv_col, _vd_col = st.columns([1.3, 1])

    with _mv_col:
        st.markdown('<div class="section-header">Movement Profile (All Pitches)</div>', unsafe_allow_html=True)
        fig_mv = go.Figure()
        fig_mv.add_hline(y=0, line_color='#2a3348', line_dash='dot', line_width=1)
        fig_mv.add_vline(x=0, line_color='#2a3348', line_dash='dot', line_width=1)

        _seen_pt = {}
        for _p in pitches:
            pg = _p['pitch_group']
            _pcolor = PT_COLOR.get(pg, '#e8eaf0')
            _lbl = GROUP_TO_DISPLAY[pg]
            _seen_pt[pg] = _seen_pt.get(pg, 0) + 1
            if _seen_pt[pg] > 1:
                _lbl = f"{_lbl} #{_seen_pt[pg]}"
            fig_mv.add_trace(go.Scatter(
                x=[_p['hb']], y=[_p['ivb']],
                mode='markers+text',
                marker=dict(size=16, color=_pcolor, line=dict(color='#0c0f14', width=2)),
                text=[_lbl], textposition='top center',
                textfont=dict(color=_pcolor, size=10, family='Barlow Condensed'),
                name=_lbl, showlegend=True,
            ))

        _lim = max(28, max(abs(_p['hb']) for _p in pitches) + 5,
                       max(abs(_p['ivb']) for _p in pitches) + 5)
        fig_mv.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#141820',
            font=dict(color='#e8eaf0', family='Barlow'),
            xaxis=dict(range=[-_lim, _lim], gridcolor='#1c2230', zeroline=False,
                       title='Horizontal Break (in) · ← Glove-Side | Arm-Side →', tickfont=dict(size=10)),
            yaxis=dict(range=[-_lim, _lim], gridcolor='#1c2230', zeroline=False,
                       title='Induced Vertical Break (in)', tickfont=dict(size=10)),
            legend=dict(bgcolor='#141820', bordercolor='#2a3348', borderwidth=1,
                        font=dict(size=10), x=0.01, y=0.99),
            margin=dict(l=0, r=0, t=10, b=0), height=400,
        )
        st.plotly_chart(fig_mv, use_container_width=True)

    with _vd_col:
        st.markdown('<div class="section-header">Velocity Differential</div>', unsafe_allow_html=True)
        _fb_ref = next((p for p in pitches if p['pitch_group'] == 'Fastball'), None) or \
                  next((p for p in pitches if p['pitch_group'] == 'Sinker'), None)
        if _fb_ref and len(pitches) > 1:
            _vd_rows = ""
            for _p in pitches:
                if _p is _fb_ref:
                    continue
                _diff = _fb_ref['velo'] - _p['velo']
                _pcolor = PT_COLOR.get(_p['pitch_group'], '#e8eaf0')
                if _p['pitch_group'] == 'Changeup':
                    if _diff >= 10:
                        _vd_ctx, _vdc = f"Pronator-typical (≥10 mph)", '#3dcc7c'
                    elif _diff >= 7:
                        _vd_ctx, _vdc = f"Average range", '#f0c040'
                    else:
                        _vd_ctx, _vdc = f"Tight — check pronation", '#f05050'
                elif _p['pitch_group'] in ('Slider', 'Sweeper'):
                    if 5 <= _diff <= 8:
                        _vd_ctx, _vdc = f"Supinator-typical (5–8 mph)", '#3dcc7c'
                    elif _diff > 8:
                        _vd_ctx, _vdc = f"Wide — pronator-like spacing", '#f0c040'
                    else:
                        _vd_ctx, _vdc = f"Very tight — hitters may recognize", '#f05050'
                else:
                    _vd_ctx, _vdc = f"Δ {_diff:.0f} mph off fastball", '#8a94aa'
                _vd_rows += f"""
                <tr style="border-bottom:1px solid #1c2230;">
                  <td style="padding:7px 12px;color:{_pcolor};font-weight:700;font-family:'Barlow Condensed',sans-serif;font-size:0.9rem;">{GROUP_TO_DISPLAY[_p['pitch_group']]}</td>
                  <td style="padding:7px 10px;color:#e8eaf0;">{_p['velo']:.1f}</td>
                  <td style="padding:7px 10px;color:{_vdc};font-weight:700;">−{_diff:.0f}</td>
                  <td style="padding:7px 10px;color:#8a94aa;font-size:0.74rem;">{_vd_ctx}</td>
                </tr>"""
            _fb_color = PT_COLOR.get(_fb_ref['pitch_group'], '#e8eaf0')
            st.markdown(f"""
            <div style="background:#141820;border:1px solid #2a3348;border-radius:6px;overflow:hidden;">
              <div style="background:#1c2230;padding:6px 12px;color:#f0c040;font-size:0.72rem;
                          font-weight:700;text-transform:uppercase;letter-spacing:0.07em;">
                FB baseline: <span style="color:{_fb_color};">{GROUP_TO_DISPLAY[_fb_ref['pitch_group']]} {_fb_ref['velo']:.1f} mph</span>
              </div>
              <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
                <thead><tr style="background:#1c2230;">
                  <th style="padding:6px 12px;text-align:left;color:#f0c040;font-size:0.68rem;text-transform:uppercase;">Pitch</th>
                  <th style="padding:6px 10px;text-align:left;color:#f0c040;font-size:0.68rem;text-transform:uppercase;">Velo</th>
                  <th style="padding:6px 10px;text-align:left;color:#f0c040;font-size:0.68rem;text-transform:uppercase;">Δ</th>
                  <th style="padding:6px 10px;text-align:left;color:#f0c040;font-size:0.68rem;text-transform:uppercase;">Context (Tread)</th>
                </tr></thead>
                <tbody>{_vd_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)
        elif _fb_ref is None:
            st.markdown("<div style='color:#505a70;font-size:0.82rem;margin-top:0.5rem;'>Add a Fastball or Sinker to enable velocity differential analysis.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#505a70;font-size:0.82rem;margin-top:0.5rem;'>Add secondary pitches to see velocity differentials.</div>", unsafe_allow_html=True)

    # ── Red flags ───────────────────────────────────────────────────────────
    _rf = []
    _fb2 = next((p for p in pitches if p['pitch_group'] in ('Fastball', 'Sinker')), None)
    _ch2 = next((p for p in pitches if p['pitch_group'] == 'Changeup'), None)
    _sl2 = next((p for p in pitches if p['pitch_group'] in ('Sweeper', 'Slider')), None)
    if _fb2 and _ch2 and _ch2['spin_eff'] > _fb2['spin_eff'] + 2:
        _rf.append(f"⚠️ Changeup spin efficiency ({_ch2['spin_eff']}%) exceeds fastball ({_fb2['spin_eff']}%) — Tread red flag: supinator side-spinning changeup")
    if _ch2 and _ch2['hb'] < 4:
        _rf.append(f"⚠️ Changeup arm-side fade is low ({_ch2['hb']:+.1f} in) — may lack deception or fight natural pronation")
    if _sl2 and ars_wrist == 'Pronator' and abs(_sl2['hb']) > 15:
        _rf.append(f"⚠️ Large horizontal break ({abs(_sl2['hb']):.0f} in) on {GROUP_TO_DISPLAY[_sl2['pitch_group']]} unusual for Pronator — verify grip (natural pronator slider has <10 in glove-side)")
    if _rf:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Red Flags</div>', unsafe_allow_html=True)
        for _r in _rf:
            st.markdown(
                f"<div style='background:#1c1014;border:1px solid #5a2020;border-left:3px solid #f05050;"
                f"border-radius:4px;padding:0.6rem 1rem;margin-bottom:0.4rem;color:#e8eaf0;"
                f"font-size:0.82rem;'>{_r}</div>",
                unsafe_allow_html=True)

    # ── Per-pitch expanders ─────────────────────────────────────────────────
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Per-Pitch Shape Analysis</div>', unsafe_allow_html=True)

    def _metric_card(col, label, value, sub, color='#f0c040'):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.5rem;color:{color};">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    _in_ars_types = {p['pitch_group'] for p in pitches}
    for _pi, _p in enumerate(pitches):
        pg = _p['pitch_group']
        _pcolor = PT_COLOR.get(pg, '#e8eaf0')
        bv  = VELO_BENCHMARKS.get(pg, {})
        bi  = IVB_BENCHMARKS.get(pg, {})
        bh  = HB_BENCHMARKS.get(pg, {})
        bsp = PITCH_SPIN_AXIS_IDEAL.get(pg, {})

        with st.expander(f"#{_pi+1} {GROUP_TO_DISPLAY[pg]} — {_p['velo']:.1f} mph  ·  {_p['ivb']:+.1f}\" IVB  ·  {_p['hb']:+.1f}\" HB", expanded=(_pi == 0)):
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)

            if bv:
                if _p['velo'] >= bv['elite']: _vt, _vc = "Elite", "#3dcc7c"
                elif _p['velo'] >= bv['avg']: _vt, _vc = "MLB Avg", "#8a94aa"
                else: _vt, _vc = "Below Avg", "#f05050"
                _metric_card(_pc1, "Velocity", f"{_p['velo']:.1f} mph", f"{_vt} · avg {bv['avg']}", _vc)
            else:
                _metric_card(_pc1, "Velocity", f"{_p['velo']:.1f} mph", "No benchmark")

            _metric_card(_pc2, "Active Spin", f"{_p['spin_eff']}%", f"{_p['spin']:,} rpm total")

            if bi:
                if abs(_p['ivb']) >= abs(bi['elite']): _it, _ic = "Elite", "#3dcc7c"
                elif abs(_p['ivb']) >= abs(bi['avg']): _it, _ic = "Avg", "#8a94aa"
                else: _it, _ic = "Below", "#f05050"
                _metric_card(_pc3, "IVB", f"{_p['ivb']:+.1f} in", f"{_it} · avg {bi['avg']}", _ic)
            else:
                _metric_card(_pc3, "IVB", f"{_p['ivb']:+.1f} in", "No benchmark")

            if bh:
                _hv = abs(_p['hb'])
                _hb = abs(bh.get('elite', bh.get('avg', 0)))
                if _hv >= _hb: _ht, _hc = "Elite", "#3dcc7c"
                elif _hv >= abs(bh.get('avg', 0)): _ht, _hc = "Avg", "#8a94aa"
                else: _ht, _hc = "Below", "#f05050"
                _metric_card(_pc4, "HB", f"{_p['hb']:+.1f} in", f"{_ht} · avg {bh.get('avg','?')}", _hc)
            else:
                _metric_card(_pc4, "HB", f"{_p['hb']:+.1f} in", "No benchmark")

            # Spin axis check
            if bsp:
                lo, hi = bsp['ideal_axis_range']
                _ax = _p['spin_axis']
                _in_range = lo <= _ax <= hi or (lo > hi and (_ax >= lo or _ax <= hi))
                _sa_note = bsp['desc']
                if not _in_range:
                    _gap = min(abs(_ax - lo), abs(_ax - hi))
                    _sa_note = f"{_gap:.0f}° off ideal — target {bsp['desc']}"
                    _sa_c = '#f05050'
                else:
                    _sa_c = '#3dcc7c'
                    _sa_note = f"Ideal: {bsp['desc']}"
                st.markdown(
                    f"<div style='color:{_sa_c};font-size:0.79rem;margin-top:6px;'>"
                    f"<b>Spin Axis {_ax}°</b> — {_sa_note}</div>",
                    unsafe_allow_html=True)

            # Wrist compatibility for this pitch
            _wc_label, _wc_col, _wc_note = WRIST_COMPAT.get(ars_wrist, {}).get(pg, ('—', '#8a94aa', ''))
            st.markdown(
                f"<div style='color:{_wc_col};font-size:0.79rem;margin-top:4px;'>"
                f"<b>Wrist Fit ({ars_wrist}):</b> {_wc_label} — {_wc_note}</div>",
                unsafe_allow_html=True)

            # Complement pitches missing from arsenal
            _comps = [c for c in _get_complements(pg) if c not in _in_ars_types]
            if _comps:
                st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='color:#505a70;font-size:0.72rem;font-weight:600;text-transform:uppercase;"
                    "letter-spacing:0.08em;'>Suggested Complements (not yet in arsenal)</div>",
                    unsafe_allow_html=True)
                for _cg in _comps:
                    _cw = _COMPLEMENT_WHY.get((pg, _cg), f"Creates movement contrast with {GROUP_TO_DISPLAY[pg]}.")
                    _cc = PT_COLOR.get(_cg, '#8a94aa')
                    st.markdown(
                        f"<div style='color:{_cc};font-size:0.82rem;margin-top:3px;'>"
                        f"▸ <b>{GROUP_TO_DISPLAY[_cg]}</b> — "
                        f"<span style='color:#8a94aa;'>{_cw}</span></div>",
                        unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PITCHER TYPE GUIDE
    # ═══════════════════════════════════════════════════════════════════════════
    _guide = PITCHER_TYPE_GUIDE.get(ars_wrist)
    if _guide:
        _gc = _guide['color']

        # Derive slot/velo context from inputs
        _slot_label = ('Sidearm / Low 3/4' if arm_angle < 30
                       else 'Three-Quarter' if arm_angle < 55
                       else 'High 3/4 / Overhand')
        # Use fastball velo for velo band if available
        _velo_ref = next((p['velo'] for p in pitches if p['pitch_group'] == 'Fastball'),
                         pitches[0]['velo'] if pitches else velo)
        _velo_band  = ('power (95+ mph)' if _velo_ref >= 95
                       else 'standard (88–94 mph)' if _velo_ref >= 88
                       else 'finesse (<88 mph)')
        _custom_key = ('low_slot' if arm_angle < 30
                       else 'three_quarter' if arm_angle < 55
                       else 'overhand')
        _velo_key   = 'power' if _velo_ref >= 95 else 'finesse' if _velo_ref < 88 else None

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="border-top:2px solid {_gc};padding-top:0.8rem;margin-bottom:0.5rem;">
          <span style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:800;
                       color:{_gc};letter-spacing:0.06em;text-transform:uppercase;">{ars_wrist} Pitcher Guide</span>
          <span style="color:#505a70;font-size:0.82rem;margin-left:0.8rem;">{_guide['tagline']}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 1: Type Profile ──────────────────────────────────────────
        st.markdown('<div class="section-header">Pitcher Type Profile</div>', unsafe_allow_html=True)
        _str_html = "".join(f"<li style='margin-bottom:3px;'>{s}</li>" for s in _guide['strengths'])
        _wk_html  = "".join(f"<li style='margin-bottom:3px;'>{w}</li>" for w in _guide['weaknesses'])
        prof_c1, prof_c2 = st.columns(2)
        with prof_c1:
            st.markdown(f"""
            <div style="background:#141820;border:1px solid #2a3348;border-left:4px solid {_gc};
                        border-radius:6px;padding:1.1rem 1.4rem;height:100%;">
              <div style="color:{_gc};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                          text-transform:uppercase;margin-bottom:0.6rem;">Mechanics</div>
              <p style="color:#e8eaf0;font-size:0.84rem;line-height:1.6;margin:0 0 0.9rem;">
                {_guide['description']}</p>
              <table style="width:100%;border-collapse:collapse;font-size:0.8rem;">
                <tr><td style="color:#505a70;padding:4px 12px 4px 0;white-space:nowrap;font-weight:600;
                               text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;">Arm Slot</td>
                    <td style="color:#e8eaf0;">{_guide['arm_slot']}</td></tr>
                <tr><td style="color:#505a70;padding:4px 12px 4px 0;white-space:nowrap;font-weight:600;
                               text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;">Wrist Action</td>
                    <td style="color:#e8eaf0;">{_guide['wrist_orientation']}</td></tr>
                <tr><td style="color:#505a70;padding:4px 12px 4px 0;white-space:nowrap;font-weight:600;
                               text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;">Spin Efficiency</td>
                    <td style="color:#e8eaf0;">{_guide['spin_efficiency']}</td></tr>
                <tr><td style="color:#505a70;padding:4px 12px 4px 0;white-space:nowrap;font-weight:600;
                               text-transform:uppercase;font-size:0.7rem;letter-spacing:0.08em;">Velo Impact</td>
                    <td style="color:#e8eaf0;">{_guide['velocity_impact']}</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)
        with prof_c2:
            st.markdown(f"""
            <div style="background:#141820;border:1px solid #2a3348;border-radius:6px;
                        padding:1.1rem 1.4rem;height:100%;">
              <div style="color:#3dcc7c;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                          text-transform:uppercase;margin-bottom:0.5rem;">Strengths</div>
              <ul style="color:#e8eaf0;font-size:0.82rem;line-height:1.55;padding-left:1.2rem;
                         margin:0 0 1rem;">{_str_html}</ul>
              <div style="color:#f05050;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                          text-transform:uppercase;margin-bottom:0.5rem;">Weaknesses / Risks</div>
              <ul style="color:#e8eaf0;font-size:0.82rem;line-height:1.55;padding-left:1.2rem;
                         margin:0;">{_wk_html}</ul>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 2: Recommended Arsenal ──────────────────────────────────
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Recommended Arsenal</div>', unsafe_allow_html=True)
        _ars_rows = ""
        for _p in _guide['arsenal']:
            _ars_rows += f"""
            <tr style="border-bottom:1px solid #1c2230;">
              <td style="padding:9px 12px;color:#e8eaf0;font-weight:700;
                         font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;
                         white-space:nowrap;">{_p['name']}</td>
              <td style="padding:9px 12px;text-align:center;color:#f0c040;font-weight:700;">{_p['pct']}</td>
              <td style="padding:9px 12px;text-align:center;color:#e8eaf0;">{_p['velo']}</td>
              <td style="padding:9px 12px;text-align:center;color:#e8eaf0;">{_p['spin']}</td>
              <td style="padding:9px 12px;text-align:center;color:#3a9dff;">{_p['active']}</td>
              <td style="padding:9px 12px;text-align:center;color:#e8eaf0;">{_p['hb']}</td>
              <td style="padding:9px 12px;text-align:center;color:#e8eaf0;">{_p['ivb']}</td>
              <td style="padding:9px 12px;color:#8a94aa;font-size:0.78rem;">{_p['why']}</td>
            </tr>
            <tr style="background:#0e1118;">
              <td colspan="2" style="padding:5px 12px 8px;">
                <span style="color:#505a70;font-size:0.68rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.08em;">Grip / Release</span><br>
                <span style="color:#8a94aa;font-size:0.76rem;">{_p['grip']}</span>
              </td>
              <td colspan="6" style="padding:5px 12px 8px;">
                <span style="color:#505a70;font-size:0.68rem;font-weight:700;text-transform:uppercase;
                             letter-spacing:0.08em;">Tunnels With</span><br>
                <span style="color:#8a94aa;font-size:0.76rem;">{_p['tunnels']}</span>
              </td>
            </tr>"""
        st.markdown(f"""
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:0.82rem;font-family:'Barlow',sans-serif;
                      background:#141820;border:1px solid #2a3348;border-radius:6px;">
          <thead>
            <tr style="background:#1c2230;">
              <th style="padding:9px 12px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Pitch</th>
              <th style="padding:9px 12px;text-align:center;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Usage%</th>
              <th style="padding:9px 12px;text-align:center;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Velo</th>
              <th style="padding:9px 12px;text-align:center;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Spin</th>
              <th style="padding:9px 12px;text-align:center;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Active Spin%</th>
              <th style="padding:9px 12px;text-align:center;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">HB</th>
              <th style="padding:9px 12px;text-align:center;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">IVB</th>
              <th style="padding:9px 12px;text-align:left;color:#f0c040;font-family:'Barlow Condensed',sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;">Why It Fits</th>
            </tr>
          </thead>
          <tbody style="color:#e8eaf0;">{_ars_rows}</tbody>
        </table>
        </div>
        <div style="color:#505a70;font-size:0.69rem;margin-top:0.3rem;">
          HB: positive = arm-side. IVB: positive = ride (resists gravity), negative = depth/drop.
        </div>
        """, unsafe_allow_html=True)

        # ── Section 3: Development Tips ──────────────────────────────────────
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Development Tips & Identification</div>', unsafe_allow_html=True)
        _tips_html = ""
        for _i, _tip in enumerate(_guide['dev_tips'], 1):
            _parts = _tip.split("**: ", 1)
            if len(_parts) == 2:
                _title = _parts[0].lstrip("**")
                _body  = _parts[1]
            else:
                _title = f"Tip {_i}"
                _body  = _tip
            _tips_html += f"""
            <div style="display:flex;gap:1rem;margin-bottom:0.8rem;align-items:flex-start;">
              <span style="flex-shrink:0;width:24px;height:24px;border-radius:50%;background:{_gc};
                           color:#0c0f14;font-weight:800;font-size:0.78rem;display:flex;
                           align-items:center;justify-content:center;">{_i}</span>
              <div>
                <div style="color:{_gc};font-weight:700;font-size:0.82rem;margin-bottom:1px;">{_title}</div>
                <div style="color:#8a94aa;font-size:0.82rem;line-height:1.55;">{_body}</div>
              </div>
            </div>"""
        st.markdown(f"""
        <div style="background:#141820;border:1px solid #2a3348;border-radius:6px;padding:1.2rem 1.4rem;">
          {_tips_html}
        </div>
        """, unsafe_allow_html=True)

        # ── Section 4: MLB Examples ───────────────────────────────────────────
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">MLB Examples</div>', unsafe_allow_html=True)
        _ex_cols = st.columns(len(_guide['mlb_examples']))
        for _ei, _ex in enumerate(_guide['mlb_examples']):
            with _ex_cols[_ei]:
                st.markdown(f"""
                <div style="background:#141820;border:1px solid #2a3348;border-top:3px solid {_gc};
                            border-radius:6px;padding:1.1rem 1.3rem;height:100%;">
                  <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:800;
                              color:#e8eaf0;line-height:1.1;">{_ex['name']}</div>
                  <div style="color:{_gc};font-size:0.72rem;font-weight:700;letter-spacing:0.08em;
                              text-transform:uppercase;margin:2px 0 0.7rem;">{_ex['team']}</div>
                  <div style="color:#8a94aa;font-size:0.8rem;line-height:1.55;margin-bottom:0.7rem;">
                    {_ex['note']}</div>
                  <div style="color:#505a70;font-size:0.68rem;font-weight:700;text-transform:uppercase;
                              letter-spacing:0.08em;margin-bottom:2px;">Arsenal Mix</div>
                  <div style="color:#e8eaf0;font-size:0.76rem;">{_ex['arsenal']}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Section 5: Customization ─────────────────────────────────────────
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Customization Guide</div>', unsafe_allow_html=True)
        _cust = _guide['customization']
        _slot_advice  = _cust.get(_custom_key, '')
        _velo_advice  = _cust.get(_velo_key, '') if _velo_key else ''
        _lhp_advice   = _cust.get('lhp', '') if handedness == 'L' else ''

        _cust_items = [
            ("Arm Slot Advice", f"{_slot_label} ({arm_angle}°)", _slot_advice, '#3a9dff'),
        ]
        if _velo_advice:
            _cust_items.append(("Velocity Band", _velo_band.title(), _velo_advice, '#f0c040'))
        if _lhp_advice:
            _cust_items.append(("LHP Considerations", "LHP", _lhp_advice, '#b47fff'))
        else:
            _cust_items.append(("Slot: Three-Quarter Alt" if _custom_key != 'three_quarter' else "Sidearm Alt",
                                 "",
                                 _cust.get('three_quarter' if _custom_key != 'three_quarter' else 'low_slot', ''),
                                 '#8a94aa'))

        _cust_cols = st.columns(len(_cust_items))
        for _ci, (_label, _sub, _text, _col) in enumerate(_cust_items):
            with _cust_cols[_ci]:
                st.markdown(f"""
                <div style="background:#141820;border:1px solid #2a3348;border-left:3px solid {_col};
                            border-radius:6px;padding:1rem 1.2rem;height:100%;">
                  <div style="color:{_col};font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                              text-transform:uppercase;">{_label}</div>
                  {"<div style='color:#e8eaf0;font-size:0.78rem;font-weight:600;margin:2px 0 0.5rem;'>" + _sub + "</div>" if _sub else "<div style='height:0.5rem'></div>"}
                  <div style="color:#8a94aa;font-size:0.8rem;line-height:1.55;">{_text}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Section 6: MLB Comps by Arm Slot × Wrist Bias ───────────────────
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">MLB Comps by Arm Slot × Wrist Bias</div>', unsafe_allow_html=True)

        _slot_order  = ['Overhand', 'High 3/4', 'Three-Quarter', 'Low 3/4', 'Sidearm']
        _type_order  = ['Pronator', 'Supinator', 'Neutral / Both']
        _type_colors = {'Pronator': '#3dcc7c', 'Supinator': '#f0c040', 'Neutral / Both': '#3a9dff'}
        _your_slot   = ('Sidearm' if arm_angle < 15 else
                        'Low 3/4' if arm_angle < 30 else
                        'Three-Quarter' if arm_angle < 50 else
                        'High 3/4' if arm_angle < 65 else 'Overhand')

        _comp_hdr = "<tr style='background:#1c2230;'>"
        _comp_hdr += "<th style='padding:8px 14px;text-align:left;color:#f0c040;font-family:\"Barlow Condensed\",sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;'>Arm Slot</th>"
        for _t in _type_order:
            _tc = _type_colors[_t]
            _comp_hdr += f"<th style='padding:8px 14px;text-align:left;color:{_tc};font-family:\"Barlow Condensed\",sans-serif;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em;'>{_t}</th>"
        _comp_hdr += "</tr>"

        _comp_rows = ""
        for _sl in _slot_order:
            _is_yours = (_sl == _your_slot)
            _row_style = f"background:#1a1f2a;border-left:3px solid {_gc};" if _is_yours else ""
            _comp_rows += f"<tr style='{_row_style}border-bottom:1px solid #1c2230;'>"
            _slot_lbl = _sl + (' ◀ your slot' if _is_yours else '')
            _comp_rows += f"<td style='padding:8px 14px;color:#e8eaf0;font-weight:700;font-family:\"Barlow Condensed\",sans-serif;white-space:nowrap;'>{_slot_lbl}</td>"
            for _t in _type_order:
                _comps_str = MLB_COMPS_BY_SLOT.get(_t, {}).get(_sl, '—')
                _tc = _type_colors[_t]
                _cell_style = f"color:{_tc};font-weight:600;" if (_is_yours and _t == ars_wrist) else "color:#8a94aa;"
                _comp_rows += f"<td style='padding:8px 14px;font-size:0.78rem;{_cell_style}'>{_comps_str}</td>"
            _comp_rows += "</tr>"

        st.markdown(f"""
        <div style="overflow-x:auto;margin-top:0.3rem;">
        <table style="width:100%;border-collapse:collapse;background:#141820;border:1px solid #2a3348;
                      border-radius:6px;font-family:'Barlow',sans-serif;">
          <thead>{_comp_hdr}</thead>
          <tbody>{_comp_rows}</tbody>
        </table>
        <div style="color:#505a70;font-size:0.69rem;margin-top:0.4rem;">
          ◀ = your current arm slot. Highlighted cell = your slot × detected wrist bias.
          Comps are representative examples — individual mechanics vary.
        </div>
        </div>
        """, unsafe_allow_html=True)

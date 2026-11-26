# Genre by Sound: What Separates Music Genres on Spotify?

**By:** Hanit Kumar

---

## Introduction

Everyone has a sense of what a genre "sounds like" — but can that be measured? Spotify computes audio features for
every track on its platform, such as how danceable, energetic, or acoustic it is. This project uses those features
to ask: **How distinct are music genres in terms of Spotify's audio features, and can we tell a track's genre
from how it sounds alone?** The answer matters for anyone who relies on automatic genre tags — playlist curators,
recommendation systems, and listeners discovering new music.

We focus on six genres from clearly different traditions: **classical, jazz, country, hip-hop, EDM, and metal**.
After cleaning, our dataset has **5,858 tracks** (about 975 per genre). The columns most relevant to our question
are:

| Column | Description |
|---|---|
| `track_genre` | Genre label the track was collected under |
| `danceability` | 0–1, how suitable a track is for dancing (tempo, rhythm stability, beat strength) |
| `energy` | 0–1, perceived intensity and activity |
| `acousticness` | 0–1, confidence that the track is acoustic |
| `instrumentalness` | 0–1, likelihood that the track has no vocals |
| `speechiness` | 0–1, presence of spoken words |
| `valence` | 0–1, musical positiveness (happy vs. sad) |
| `loudness` | Average loudness in decibels |
| `tempo` | Estimated beats per minute |
| `duration_ms` | Track length in milliseconds |
| `explicit` | Whether the track has explicit lyrics |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The raw dataset has 114,000 tracks across 114 genre labels. We kept only our six genres (6,000 tracks), then:

1. **Removed duplicate and ambiguous tracks.** The same song can be collected from several genre playlists. We
   dropped exact repeats within a genre, and dropped tracks listed under *more than one* of our six genres, since
   their genre label is ambiguous and they could appear in both our training and test data later. This removed
   142 tracks.
2. **Turned impossible values into missing values.** A `tempo` of 0 BPM means Spotify's beat tracker failed, not
   that the song has no tempo, so we replaced it with `NaN` (same for a `time_signature` or `duration_ms` of 0).
   This leaves about 21% of tracks with missing tempo, which we study below.
3. **Parsed release dates.** `release_date` mixes `YYYY`, `YYYY-MM`, and `YYYY-MM-DD` formats depending on how
   precisely the label recorded it, so we extracted `release_year` and `decade`.
4. **Added helper columns**: `duration_min` (easier to read than milliseconds), `n_artists`, and the primary
   artist's follower count and popularity, joined from a separate artist table by name.

The first rows of the cleaned data:

| track_name             | primary_artist   | track_genre   |   danceability |   energy |   acousticness |   tempo |   release_year |
|:-----------------------|:-----------------|:--------------|---------------:|---------:|---------------:|--------:|---------------:|
| Zara Zara              | Bombay Jayashri  | classical     |          0.643 |    0.268 |          0.593 | nan     |           2001 |
| Zara Zara - Lofi       | Bombay Jayashri  | classical     |          0.608 |    0.638 |          0.581 | 140.109 |           1984 |
| Vaseegara              | Bombay Jayashri  | classical     |          0.695 |    0.293 |          0.596 | nan     |           1972 |
| Zara Zara - LoFi Chill | Bombay Jayashri  | classical     |          0.583 |    0.308 |          0.581 | 118.226 |           1987 |
| Zara Zara              | Bombay Jayashri  | classical     |          0.643 |    0.268 |          0.593 | nan     |           2001 |

### Univariate Analysis

<iframe src="assets/univariate-danceability.html" width="800" height="600" frameborder="0"></iframe>

Danceability across our six genres is roughly bell-shaped around 0.55, but with a long tail of low values. Those
low values are mostly classical and jazz tracks — a first hint that the genres sound different.

<iframe src="assets/univariate-acousticness.html" width="800" height="600" frameborder="0"></iframe>

Acousticness is strongly bimodal: most tracks are either clearly electronic/amplified (near 0) or clearly acoustic
(near 1), with few in between. A feature that splits tracks this sharply should be useful for telling genres apart.

### Bivariate Analysis

<iframe src="assets/bivariate-danceability-by-genre.html" width="800" height="600" frameborder="0"></iframe>

Hip-hop has the highest median danceability — higher even than EDM, which is made for dancing — while classical
has the lowest. We test whether that surprising hip-hop vs. EDM gap is real in the Hypothesis Testing section.

<iframe src="assets/bivariate-energy-acousticness.html" width="800" height="600" frameborder="0"></iframe>

Energy and acousticness are strongly negatively related, and the genres form visible clusters: classical and jazz
in the acoustic, low-energy corner; metal and EDM in the electronic, high-energy corner; country and hip-hop in
between.

### Interesting Aggregates

Average audio features per genre:

| track_genre   |   danceability |   energy |   acousticness |   instrumentalness |   speechiness |   valence |
|:--------------|---------------:|---------:|---------------:|-------------------:|--------------:|----------:|
| classical     |          0.38  |    0.189 |          0.924 |              0.616 |         0.051 |     0.377 |
| jazz          |          0.51  |    0.353 |          0.715 |              0.078 |         0.055 |     0.49  |
| country       |          0.556 |    0.596 |          0.323 |              0.005 |         0.042 |     0.521 |
| hip-hop       |          0.736 |    0.683 |          0.196 |              0.011 |         0.13  |     0.551 |
| edm           |          0.648 |    0.757 |          0.115 |              0.04  |         0.07  |     0.465 |
| metal         |          0.465 |    0.841 |          0.037 |              0.064 |         0.07  |     0.418 |

Each genre has a distinct "fingerprint": classical stands out on acousticness and instrumentalness, hip-hop on
danceability and speechiness, and metal on energy. Country sits in the middle on almost every feature, which
suggests it will be the hardest genre to identify.

---

## Assessment of Missingness

### NMAR Analysis

We do not believe `tempo` is **NMAR**. Whether Spotify fails to detect a tempo should depend on how the music
sounds — quiet, free-flowing, or beatless passages are hard to track — and those qualities are largely captured by
columns we *do* observe, such as `energy`, `loudness`, and `track_genre`. The one way it could be NMAR is if the
beat tracker fails more often at certain true tempos (for example, very slow pieces). To rule that out and fully
explain the missingness, we would want Spotify's beat-tracking confidence score and per-section tempo estimates,
which aren't in this dataset.

### Missingness Dependency

We ran permutation tests (α = 0.05, 1,000 shuffles each) on whether `tempo` is missing:

- **Genre — depends.** Using total variation distance (TVD) between the genre distributions of tracks with and
  without a tempo, the observed TVD was 0.226, larger than every shuffled value (p ≈ 0.000). Classical and jazz
  are missing tempo about three times as often as metal.
- **Duration — does not depend.** The difference in mean duration between tracks with and without a tempo was
  only about 3 seconds (0.053 minutes), well within what shuffling produces (p ≈ 0.37).

<iframe src="assets/missingness-tempo-by-genre.html" width="800" height="600" frameborder="0"></iframe>

<iframe src="assets/missingness-genre-tvd.html" width="800" height="600" frameborder="0"></iframe>

Since tempo's missingness depends on genre (an observed column), it looks Missing at Random (MAR). For modeling,
this means the *fact* that tempo is missing is itself a clue about a track's genre.

---

## Hypothesis Testing

The box plot above suggested hip-hop is more danceable than EDM. We tested this formally.

- **Null hypothesis:** Hip-hop and EDM tracks have the same distribution of danceability; the observed difference
  in means is due to random chance.
- **Alternative hypothesis:** Hip-hop tracks are, on average, more danceable than EDM tracks.
- **Test statistic:** mean danceability of hip-hop minus mean danceability of EDM.
- **Significance level:** 0.05.

We used a permutation test, shuffling genre labels among the 985 hip-hop and 987 EDM tracks 10,000 times. A
permutation test fits because we are asking whether two samples look like they come from the same population, and
it doesn't assume danceability is normally distributed. We used the signed difference rather than the absolute
difference because our alternative has a direction (hip-hop > EDM).

<iframe src="assets/hypothesis-danceability.html" width="800" height="600" frameborder="0"></iframe>

**Result:** the observed difference was **0.088**, while no shuffled difference exceeded 0.018, giving a
**p-value ≈ 0.000**. We reject the null hypothesis. The data are consistent with hip-hop being more danceable on
average than EDM by Spotify's measure. This doesn't prove hip-hop is "more danceable" in general: Spotify's metric
rewards steady, strong beats, and these tracks may not represent each genre as a whole.

---

## Framing a Prediction Problem

We predict a track's **genre** (`track_genre`) from its audio features. This is a **multiclass classification**
problem with six classes. We chose this response variable because it directly tests our central question: if
genres really sound different, a model should be able to tell them apart from sound alone.

**Features at the time of prediction:** we only use features Spotify computes from the audio file itself
(danceability, energy, acousticness, speechiness, instrumentalness, valence, loudness, tempo, duration) plus the
`explicit` flag. All of these exist as soon as a track is uploaded, before anyone assigns it a genre. We do not use
popularity or artist follower counts, which build up after release and describe the artist rather than the sound.

**Metric: macro-averaged F1-score.** Our classes are nearly balanced, so accuracy wouldn't be badly misleading,
but macro F1 gives every genre equal weight and balances precision and recall. That stops a model from scoring well
by nailing the easy genres (classical, metal) while confusing the hard ones (country, jazz). We report accuracy
alongside it.

---

## Baseline Model

Our baseline is a single sklearn `Pipeline` with a **multinomial logistic regression** classifier, trained on 75%
of the data and tested on the remaining 25% (split stratified by genre). It uses five features:

- **4 quantitative:** `danceability`, `energy`, `acousticness`, `speechiness` — standardized with
  `StandardScaler`, since logistic regression is sensitive to feature scale.
- **1 nominal:** `explicit` (True/False) — one-hot encoded into a single 0/1 column.
- **0 ordinal.**

| Split | Macro F1 | Accuracy |
|---|---|---|
| Train | 0.600 | 0.606 |
| Test | 0.601 | 0.605 |

<iframe src="assets/baseline-confusion-matrix.html" width="800" height="600" frameborder="0"></iframe>

The baseline scores far above random guessing (about 0.17 for six balanced genres), and its train and test scores
match, so it isn't overfitting. Classical and metal are easiest to identify; country and jazz are hardest, because
they sit in the middle on every feature the model uses. We don't consider this model "good" yet: about 4 in 10
test tracks are mislabeled. A linear model on four features can't capture the clustered, non-linear patterns we
saw in our scatter plot, and it ignores clearly useful features such as instrumentalness and loudness.

---

## Final Model

*Coming soon.*

---

## Fairness Analysis

*Coming soon.*

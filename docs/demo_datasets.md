# Demo datasets: what we may use, and why the licence field is not evidence

For examples, screenshots, videos and other project materials. The rule is the
one in `docs/model_license_policy.md`: redistribution requires evidence from
the actual rights holder, so "the page said CC0" is not a defence.

## The finding that governs everything below

**Roboflow Universe's licence field is asserted by the uploader and verified by
nobody.** Two documented cases where it is simply wrong:

- `popular-benchmarks/mit-indoor-scene-recognition` displays **`License: MIT`**
  — a software licence stamped onto MIT-67, whose own authors state _"The images
  provided here are for research purposes only"_ and which was _"collected from
  Google, Altavista, Flickr and the LabelMe data set."_
- `handrecognitionpro/hagrid_dataset` displays **CC BY 4.0** while upstream
  HaGRID is CC BY-**SA** 4.0.

And of the five Roboflow-curated datasets we examined, **three carry a licence
that differs from their upstream**: CC BY 2.0 shown as 4.0, MIT shown as CC0,
and "Unknown" shown as Public Domain.

So the check is never "what does the page say". It is: _who made the pixels, and
what did they say?_ Querying Roboflow's API confirms what the uploader typed, not
what they had the right to type.

## Approved

| dataset                                                       | licence, and the real one                                                             | why it is safe                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `joseph-nelson/chess-pieces-new`                              | Public Domain, first-party                                                            | **Safest asset we have.** Roboflow photographed these themselves, from a tripod, 289 images. No upstream, no people in frame, rights holder and licensor are the same party.                                                                                                  |
| `joseph-nelson/fruits-dataset`                                | Universe says CC0; **upstream is MIT**, © 2017-2020 Mihai Oltean & Horea Muresan      | Fruits-360. First-party photography — fruit on a 3 rpm motor against white paper. Commercially fine, but **ship the MIT notice**; CC0 waives exactly the attribution MIT requires.                                                                                            |
| `david-lee-d0rhs/american-sign-language-letters`              | Public Domain (CC0 1.0), confirmed twice                                              | **The hand pick.** The only hand dataset on Universe with documented capture: shot at 720p/1080p deliberately "to mirror the intended environment on a mobile device or webcam", one hand per frame, plain background. Which is exactly what mediapipe wants.                 |
| `potatocare/potatocare-multi-class-2`                         | Universe says MIT; class names are verbatim PlantVillage, and **PlantVillage is CC0** | 2,155 images, semantic segmentation, no people. The underlying imagery is genuinely public domain even though the label disagrees.                                                                                                                                            |
| `suhjeong-kim-tjme8/food_semantic_segmentation`               | Public Domain                                                                         | Only semseg candidate with a real description of its own labelling. 148 images, food vs plate — small, but legible in a screenshot.                                                                                                                                           |
| `korea-maritime-ocean-university/semantic-segmentation-dqanf` | Public Domain                                                                         | 267 images, ship/bridge/buoy/land/sea/sky. Undescribed, but CC0 and no people.                                                                                                                                                                                                |
| `joseph-nelson/rock-paper-scissors`                           | Universe says CC BY 4.0; **Laurence Moroney says CC BY 2.0**                          | **CGI, not photographs** — he rendered them, so there is no third-party image-rights layer at all. Attribution to Laurence Moroney is mandatory and Roboflow's page does not mention it. Spike mediapipe on ~20 frames first: CGI hands are not guaranteed to land landmarks. |

## Rejected, and why

- **`joseph-nelson/flowers-tzb4f`** — Universe says Public Domain. Upstream is a
  Kaggle set whose licence field reads **"Unknown"** and whose author states he
  _"scraped data from flickr, google images, and yandex images"_. Roboflow is
  redistributing under CC0 something nobody ever CC0'd. Not usable at any
  confidence.
- **`roboflow-58fyf/rock-paper-scissors-sxsw`** — built from four sources, one of
  which is the MIT-67 mirror above (research-only, scraped). It also contains
  identifiable SXSW attendees, adding personality rights on top of a licence
  problem. No licence string appears on its page at all.
- **`joseph-nelson/hard-hat-workers`** — this one is a judgement call, not a
  clear no. Upstream CC0 is genuine: the Harvard Dataverse DOI's DataCite record
  really does say `cc0-1.0`. But the associated paper (Sensors 20(7):1868) never
  says how the images were obtained, neighbouring datasets in that literature
  scraped Google and Baidu explicitly, and the subjects are identifiable
  workers. Fine for our own testing, which is what we already use it for.
  **Not for an advertisement with a recognisable face in it.**
- **`roboflow-universe-projects/windows-segmentation`**,
  `roboflow-jvuqo/fashion-assistant-segmentation`,
  `roboflow-universe-projects/fire-and-smoke-segmentation` — all display CC BY
  4.0 and all say _"A description for this project has not been published yet."_
  Architectural photos, clothing shots of people, and fire imagery are precisely
  the three categories where stock and web sourcing is the norm, and there is
  nothing to rebut it. Treat as unknown provenance.
- **`handrecognitionpro/hagrid_dataset`** — licence conflict above, and the frames
  are upper-body at 0.5-4 m, so the hand is small. Wrong shape for us anyway.
- **Two-handed sign languages** (Indian SL, BISINDO) and gesture sets whose
  classes include Clap, Handshake, Heart, Namaste, Prayer — our handpose pipeline
  reads one hand per image.

## Approved for engineering validation, not redistribution

- **Spondylolisthesis Vertebral Landmark v1** — 698 real sagittal X-rays with
  four corner keypoints per vertebra, published by Karla Reyes under CC BY 4.0
  at DOI `10.17632/5jdfdgp762.1`. It is an unusually good functional test for
  generic multi-instance keypoints and is documented in
  `docs/keypoint_realworld_validation.md`. We download it directly and retain
  attribution; we do not bundle it or use medical imagery in marketing. The
  dataset description identifies two underlying sources, so redistribution of
  the pixels gets a separate third-party-rights review despite the dataset-level
  license.

## A trap worth knowing

`roboflow-100/sign-language-sokdr` and `asl-dataset/asl-alphabet-dataset` are
almost certainly **the same 720 images** as the David Lee set — its page even
says "Originally created by David Lee". Using more than one silently triples the
same data and makes a validation split meaningless.

## Where the licence differs from upstream, we follow upstream

Stated plainly because it decides several rows above: when Roboflow's label is
_more_ permissive than the source (CC0 over MIT, Public Domain over "Unknown"),
the source wins and we either honour the stricter terms or drop the dataset.
When it is _less_ permissive there is nothing to do — the stricter label is the
one we can rely on.

## Method, for whoever repeats this

`universe.roboflow.com` returns 403 to a plain fetch. Reading a page through
`https://r.jina.ai/https://universe.roboflow.com/<workspace>/<slug>` renders past
it and returns the licence literally, as `Task:<type>License:<licence>`. The
`?license=` search parameter does **not** filter — it returns the identical list
— so a licence has to be read per project, one page at a time. That is the whole
reason this took a day rather than an hour.

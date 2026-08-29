# Create a new trainer

There are two shapes of this, and which one you are doing changes almost
everything below.

**A new project type** — a task the application cannot do at all. Follow the
whole document: the type string is a new join key between the creation form,
`MODEL_VARIANTS` and `TrainerBuilder`.

**A second model for a project type that already exists** — RF-DETR alongside
NanoDet, for instance. Skip step 1 entirely. Do not add a project type: a
detection project is a detection project whichever architecture trains it, and
splitting it in two would strand every export, import and label a user already
has. What you add is variants in `MODEL_VARIANTS` with a new
`model_architecture`, and an entry in `ALTERNATIVE_TRAINERS` (step 3) so the
builder can tell the two apart. Then check the four places that assumed one
trainer per type:

- `TrainerBuilder.get_trainer_class` takes the architecture as a second
  argument. `routers/model.py` passes the **model's** architecture when running
  inference, `run_training_job` passes the **run's**.
- `/api/augmentations` answers per `"<type>::<architecture>"` as well as per
  type, because two models under one type rarely augment the same way.
- The training dialog filters the starting-model list by architecture and size
  already, so a user cannot pick a NanoDet checkpoint to fine-tune an RF-DETR.
  A direct API call still can, so validate it in the trainer.
- `docs/model_license_policy.md` — a second model means a second set of weights
  in the installer, with its own licence.

## 1. Add a new project type

- Add a new project type to `frontend/src/components/project-creation-form.tsx`.
- Add the same string to `MODEL_VARIANTS` in `anylearning/config.py`, with the
  variants the UI should offer.
- Example project types: `Object Detection` and `Image Segmentation`.

The project-type string is the join key between the form, `MODEL_VARIANTS` and
`TrainerBuilder.get_trainer_class`. If the three disagree, project creation
succeeds and training fails later with a confusing lookup error, so change them
together. `tests/e2e/test_all_flows_e2e.py` asserts the variant lists match
`config.MODEL_VARIANTS`, and will fail if you add a variant without covering
it — its assertions are scoped to the architecture each test goes on to train,
so a second architecture under an existing type needs its own file rather than
a longer list here.

## 2. Add a new trainer class

- Add a new trainer class to `anylearning/training/trainers`.
- Example trainer classes: `NanoDetTrainer` and `SemSegTrainer`.
- Inherit from `BaseTrainer`.
- See `anylearning/training/training_job.py` for the order of function calls:
  `prepare_data` → `prepare_config` → `train` → `export_onnx` → `get_model_path`.

Two parts of that contract are easy to get wrong:

- **`export_onnx` gates registration.** The model row is only written after it
  succeeds, so an export that raises throws away a finished training run.
  Return `None` if the trainer genuinely has no ONNX to offer — that is treated
  as a warning, not a failure.
- **`run_inference` is a `@staticmethod`**, called by `routers/model.py` without
  constructing a trainer. Keep it free of instance state; constructing a trainer
  requires a live database.
- Use `self.resolve_pretrained_model_path()` rather than reading
  `training_params.pretrained_model` directly. It returns the checkpoint path or
  `None`, and handles the values that are not model ids.

- If the model needs pretrained weights, they have to **ship**. The application
  is sold on training with no network, and a library that downloads a backbone
  on first use fails after the dataset has already been exported, inside a
  subprocess, where the user sees a run that stopped. Add a step to
  `fetch_weights.py`, point the library's cache at the bundled directory in
  `anylearning/weights.py`, and make the trainer _raise_ when the file is
  missing rather than fall back — several loaders load with `strict=False`,
  which turns a missing checkpoint into a model that trains from random
  initialisation and looks fine until the metrics arrive.

## 3. Add a new trainer builder

- Add a new trainer builder to `anylearning/training/trainers/trainer_builder.py`.
- One trainer per project type goes in `DEFAULT_TRAINERS`; a second model for a
  type that already has one goes in `ALTERNATIVE_TRAINERS`, keyed by
  `(project type, model_architecture)`.
- Import a heavy dependency lazily if only that trainer needs it. This module is
  imported by the API process at startup, so a module-level `import` there is
  paid on every launch and in every spawned training child -- `import rfdetr`
  alone costs about 1.8 seconds. Add the module to `TRAINER_MODULES` in
  `routers/health.py` as well, and the third-party package beside it: a lazy
  import means `/api/health/imports` learns nothing about it otherwise, and a
  packaged build that dropped it would say so nowhere.

## 4. Test the new trainer

- Upload data and test the new trainer.
- The training data will be located in `~/anylearning-data/projects/<project_id>/training/<training_session_id>`.

## Debugging tips

- Use `--development` flag to run the server in development mode. This will not delete the training folder after the training job is done.
- If seeing database errors, try removing folder `~/anylearning-data/` and re-run the server. The database will be reset!!! (This is a known issue because the migration is not idempotent.)
- Check the prepared data in the training folder: `~/anylearning-data/projects/<project_id>/training/<training_session_id>/data/`.

## For image classification

- The class id should be between 0 and `num_classes - 1`.
- A licence-cleared sample is available in the
  [ZhangLabData chest X-ray folder](https://huggingface.co/datasets/nrl-ai/anylearning-data/tree/main/zhanglabdata_chest_xray).
  Use `train.zip` for training and `test.zip` for validation and testing. The
  dataset is CC BY 4.0; retain its attribution when redistributing it.
- Steps to import as labeled data:
  - Create all classes in the project settings:
    - NORMAL
    - PNEUMONIA
  - Import the data from zip files.

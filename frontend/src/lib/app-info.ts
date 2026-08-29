/**
 * Version, in one place on the frontend.
 *
 * `anylearning/app_info.py` is the project-wide source of truth; this mirrors
 * it for the UI. Keep the two in step -- the version used to appear in six
 * places and nothing caught a mismatch.
 */
export const APP_VERSION = "0.26.3"

/**
 * The release name, mirroring `__product__` in `anylearning/app_info.py`.
 *
 * This is display text only; the open-source build has no activation tier.
 */
export const PRODUCT_NAME = "AnyLearning"

/**
 * Where the app keeps everything, from `anylearning/config.py`.
 *
 * There is no setting for this: `DATA_ROOT` is derived from the home directory
 * with no override, so showing it is reporting a fact rather than offering a
 * choice.
 */
export const DATA_ROOT = "~/anylearning-data"

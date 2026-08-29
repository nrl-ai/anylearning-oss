# Auto Labeling Format

## Request

POST `/api/projects/{project_id}/auto_labeling/inference`

Example request body:

```json
{
  "model_name": "sam",
  "data_item_id": 1261,
  "marks": [
    { "data": [78, 53], "label": 1, "type": "point" },
    { "data": [115, 56], "label": 0, "type": "point" },
    { "data": [178, 50, 222, 102], "label": 1, "type": "rectangle" }
  ],
  "preload_data_item_ids": [1262, 1263, 1264]
}
```

- `model_name`: Pick from the list of models in the auto labeling UI.
- `data_item_id`: The data item id to label.
- `marks`: The marks to label.
  - For `point`: `label = 1` is positive, `label = 0` is negative.
  - For `rectangle`: `label = 1` is positive, `label = 0` is negative.
  - `data` is a list of coordinates. A point is `[x, y]`; a rectangle is `[x1, y1, x2, y2]`.
- `preload_data_item_ids`: The data item ids to preload.

The response contains SAM mask contours. The labeling client preserves those
contours in Image Segmentation projects and converts them to their four-corner
bounding boxes in Object Detection projects. The annotation API also rejects
non-rectangle shapes for Object Detection, so a client cannot accidentally save
segmentation geometry into a detection dataset.

"""The dataset API: upload, annotation, export.

This is the largest router and the one the labelling UI talks to constantly.
The tests below drive it through real HTTP, so they cover request validation,
status codes and the response_model serialisation -- and they focus on the
failure paths, because those are what a user hits when a project is deleted
mid-session or an upload is retried.
"""

import io
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def api(tmp_path, monkeypatch):
    from anylearning import config, database

    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", str(projects_root))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "main.db"))
    # upload_data spools the archive into DATA_ROOT/tmp_files. Left unpatched,
    # a test run writes into the developer's own ~/anylearning-data.
    monkeypatch.setattr(config, "DATA_ROOT", str(tmp_path))

    manager = database.DatabaseManager()
    monkeypatch.setattr(database, "db_manager", manager)

    from anylearning.routers import dataset, project

    for module in (dataset, project):
        monkeypatch.setattr(module, "db_manager", manager, raising=False)

    app = FastAPI()
    app.include_router(project.router)
    app.include_router(dataset.router)
    with TestClient(app) as client:
        client.projects_root = projects_root
        yield client
    manager.dispose_all()


@pytest.fixture
def project_id(api):
    response = api.post(
        "/api/projects",
        json={"name": "Data", "type": "Image Classification", "description": ""},
    )
    assert response.status_code == 200, response.text
    return response.json()["id"]


def png_bytes(size=(16, 16), colour=(255, 0, 0)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _zip_with_images(count=2):
    """A ZIP of PNGs, which is the shape upload_data actually expects."""
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(count):
            archive.writestr(f"image_{index}.png", png_bytes().getvalue())
    buffer.seek(0)
    return buffer


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------


def test_data_items_of_a_new_project_is_empty(api, project_id):
    response = api.get(f"/api/projects/{project_id}/data_items")
    assert response.status_code == 200
    body = response.json()
    assert body["data_items"] == []
    assert body["total_count"] == 0


def test_data_items_echo_pagination(api, project_id):
    response = api.get(f"/api/projects/{project_id}/data_items?offset=0&limit=10")
    body = response.json()
    assert body["offset"] == 0
    assert body["limit"] == 10


def test_subset_filter_is_optional(api, project_id):
    """subset is Optional[int] = None; omitting it must not 422."""
    assert api.get(f"/api/projects/{project_id}/data_items").status_code == 200


@pytest.mark.parametrize("subset", [0, 1, 2])
def test_subset_filter_accepts_each_split(api, project_id, subset):
    response = api.get(f"/api/projects/{project_id}/data_items?subset={subset}")
    assert response.status_code == 200


def test_subset_filter_rejects_a_non_integer(api, project_id):
    response = api.get(f"/api/projects/{project_id}/data_items?subset=train")
    assert response.status_code == 422


def test_datasets_endpoint_responds_for_a_new_project(api, project_id):
    assert api.get(f"/api/projects/{project_id}/datasets").status_code == 200


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


def test_upload_status_before_any_upload(api, project_id):
    """The UI polls this immediately; it must not 500 on a fresh project."""
    response = api.get(f"/api/projects/{project_id}/upload_status")
    assert response.status_code in (200, 404)


def test_upload_rejects_a_request_with_no_files(api, project_id):
    response = api.post(f"/api/projects/{project_id}/upload_data")
    assert response.status_code == 422


def test_upload_accepts_a_bare_image(api, project_id):
    """Images no longer have to be zipped first.

    Making a .zip to add three photographs is busywork the app can do itself,
    and it was the most-reported friction in uploading.
    """
    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("first.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text

    items = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"]
    assert [item["original_name"] for item in items] == ["first.png"]


def test_upload_accepts_several_images_at_once(api, project_id):
    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files=[
            ("file", ("a.png", png_bytes(), "image/png")),
            ("file", ("b.png", png_bytes(), "image/png")),
        ],
    )
    assert response.status_code == 200, response.text

    items = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"]
    assert sorted(item["original_name"] for item in items) == ["a.png", "b.png"]


def test_upload_rejects_a_file_that_is_neither(api, project_id):
    """Named in the message: "unsupported" leaves the user guessing which one."""
    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
    assert ".txt" in response.json()["detail"]


def test_upload_rejects_a_zip_mixed_with_images(api, project_id):
    """Ambiguous: the archive would be repacked inside another archive."""
    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files=[
            ("file", ("data.zip", _zip_with_images(), "application/zip")),
            ("file", ("a.png", png_bytes(), "image/png")),
        ],
    )
    assert response.status_code == 400


def test_upload_accepts_a_zip(api, project_id):
    zip_bytes = _zip_with_images()
    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("data.zip", zip_bytes, "application/zip")},
    )
    # Accepted or queued -- either way it must not be a validation failure.
    assert response.status_code != 422, response.text
    assert response.status_code < 500, response.text


def _zip_with_polygon_labels(label="cell"):
    """A ZIP shaped like an exported AnyLabeling project: image + its .json."""
    import json
    import zipfile

    annotation = {
        "version": "0.3.3",
        "shapes": [
            {
                "label": label,
                "points": [[2, 2], [12, 2], [12, 12], [2, 12]],
                "shape_type": "polygon",
            }
        ],
        "imagePath": "sample.png",
        "imageHeight": 16,
        "imageWidth": 16,
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sample.png", png_bytes().getvalue())
        archive.writestr("sample.json", json.dumps(annotation))
    buffer.seek(0)
    return buffer


def test_upload_creates_project_labels_from_polygon_annotations(api):
    """A segmentation upload brings its own categories with it.

    Uploading an annotated set and then finding no labels on the project meant
    every polygon in it was orphaned: the labelling UI had nothing to colour
    them with and training had no classes to learn. Categories come from the
    annotations themselves here, so this holds regardless of the
    `auto_create_categories` flag -- that flag is about folder names, which a
    segmentation set does not have.
    """
    created = api.post(
        "/api/projects",
        json={"name": "Cells", "type": "Image Segmentation", "description": ""},
    )
    assert created.status_code == 200, created.text
    project_id = created.json()["id"]

    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("data.zip", _zip_with_polygon_labels(), "application/zip")},
    )
    assert response.status_code == 200, response.text

    labels = api.get(f"/api/projects/{project_id}").json()["labels"]
    assert [label["name"] for label in labels] == ["cell"]

    items = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"]
    assert len(items) == 1
    assert items[0]["labeled"] is True


def test_class_distribution_of_an_empty_project(api, project_id):
    body = api.get(f"/api/projects/{project_id}/class_distribution").json()
    assert body["classes"] == []
    assert body["unlabeled"]["total"] == 0


def test_class_distribution_counts_annotations_per_subset(api):
    """Counts come from the shapes, not from the item count.

    An image with three boxes on it is three annotations, and that is the
    number that decides whether a class is under-represented -- counting
    images instead reports a class as healthy when one image carries all of it.
    """
    created = api.post(
        "/api/projects",
        json={"name": "Counts", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]

    api.post(
        f"/api/projects/{project_id}/upload_data",
        params={"subset": 0},
        files={
            "file": ("train.zip", _zip_with_polygon_labels("cat"), "application/zip")
        },
    )
    api.post(
        f"/api/projects/{project_id}/upload_data",
        params={"subset": 1},
        files={"file": ("val.zip", _zip_with_polygon_labels("dog"), "application/zip")},
    )

    body = api.get(f"/api/projects/{project_id}/class_distribution").json()
    by_name = {row["name"]: row for row in body["classes"]}

    assert by_name["cat"]["train"] == 1
    assert by_name["cat"]["validation"] == 0
    assert by_name["dog"]["validation"] == 1
    # dog appears only in validation: it is never trained on, which is exactly
    # the imbalance this endpoint exists to make visible.
    assert by_name["dog"]["train"] == 0
    assert all(row["known"] for row in body["classes"])


def test_class_distribution_reports_a_category_the_project_no_longer_lists(api):
    """A renamed or deleted label leaves annotations behind that still train."""
    created = api.post(
        "/api/projects",
        json={"name": "Orphans", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]

    api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("d.zip", _zip_with_polygon_labels("ghost"), "application/zip")},
    )
    # ProjectUpdate.labels is a pydantic `Json` field, so the value travels as
    # a JSON *string* rather than as a list.
    cleared = api.patch(f"/api/projects/{project_id}", json={"labels": "[]"})
    assert cleared.status_code == 200, cleared.text

    body = api.get(f"/api/projects/{project_id}/class_distribution").json()
    ghost = next(row for row in body["classes"] if row["name"] == "ghost")
    assert ghost["known"] is False
    assert ghost["total"] == 1


@pytest.mark.parametrize("subset", [-1, 3, 99])
def test_upload_rejects_an_out_of_range_subset(api, project_id, subset):
    """subset indexes [train, val, test]; anything else is a client error."""
    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        params={"subset": subset},
        files={"file": ("data.zip", _zip_with_images(), "application/zip")},
    )
    assert response.status_code == 400


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------


def test_get_annotation_for_missing_item_is_404(api, project_id):
    response = api.get(f"/api/projects/{project_id}/data_items/999/get_annotation")
    assert response.status_code == 404


def test_set_annotation_on_missing_item_is_404(api, project_id):
    response = api.post(
        f"/api/projects/{project_id}/data_items/999/set_annotation",
        json={"data": []},
    )
    assert response.status_code in (404, 422)


def test_object_detection_annotation_saves_and_reloads_boxes(api):
    created = api.post(
        "/api/projects",
        json={"name": "Detection", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]
    uploaded = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("worker.png", png_bytes(), "image/png")},
    )
    assert uploaded.status_code == 200, uploaded.text
    item_id = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"][0][
        "id"
    ]
    boxes = [
        {
            "type": "rectangle",
            "points": [[2, 3], [2, 12], [14, 12], [14, 3]],
            "categories": ["helmet"],
            "phi": 0,
        }
    ]

    saved = api.post(
        f"/api/projects/{project_id}/data_items/{item_id}/set_annotation",
        json=boxes,
    )

    assert saved.status_code == 200, saved.text
    reloaded = api.get(
        f"/api/projects/{project_id}/data_items/{item_id}/get_annotation"
    )
    assert reloaded.status_code == 200
    assert reloaded.json() == boxes


def test_object_detection_annotation_rejects_polygons(api):
    created = api.post(
        "/api/projects",
        json={"name": "Detection", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]
    uploaded = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("worker.png", png_bytes(), "image/png")},
    )
    assert uploaded.status_code == 200, uploaded.text
    item_id = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"][0][
        "id"
    ]

    response = api.post(
        f"/api/projects/{project_id}/data_items/{item_id}/set_annotation",
        json=[
            {
                "type": "polygon",
                "points": [[2, 3], [14, 3], [8, 12]],
                "categories": ["helmet"],
            }
        ],
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Object Detection annotations must contain rectangles only"
    )


@pytest.mark.parametrize(
    "points",
    [
        [[2, 3], [14, 3], [8, 12]],
        [[2, 3], [2, 3], [2, 3], [2, 3]],
        [[2, 3], [4, 6], [6, 9], [8, 12]],
        [[2, 3], [2, 12], [14, 12], ["fourteen", 3]],
    ],
)
def test_object_detection_annotation_rejects_malformed_rectangles(api, points):
    created = api.post(
        "/api/projects",
        json={"name": "Detection", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]
    uploaded = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("worker.png", png_bytes(), "image/png")},
    )
    assert uploaded.status_code == 200, uploaded.text
    item_id = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"][0][
        "id"
    ]

    response = api.post(
        f"/api/projects/{project_id}/data_items/{item_id}/set_annotation",
        json=[{"type": "rectangle", "points": points, "categories": ["helmet"]}],
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Object Detection annotations must contain rectangles only"
    )


def test_set_class_id_on_missing_item_is_404(api, project_id):
    response = api.post(
        f"/api/projects/{project_id}/data_items/999/class_id",
        json={"class_id": 1},
    )
    assert response.status_code in (404, 422)


def test_relabel_handpose_updates_landmark_target_without_losing_metadata(api):
    created = api.post(
        "/api/projects",
        json={"name": "Hands", "type": "Handpose Classification", "description": ""},
    )
    project_id = created.json()["id"]
    api.patch(
        f"/api/projects/{project_id}",
        json={
            "labels": json.dumps(
                [
                    {"name": "open", "color": "#fff", "id": 0},
                    {"name": "fist", "color": "#000", "id": 3},
                ]
            )
        },
    )

    from sqlalchemy.orm import Session

    from anylearning.database import DataItem, Dataset, db_manager

    landmarks = {
        str(index): {"x": index / 21, "y": 0.5, "z": 0.0} for index in range(21)
    }
    with Session(db_manager.get_project_engine(project_id)) as session:
        dataset = Dataset()
        session.add(dataset)
        session.flush()
        item = DataItem(
            dataset_id=dataset.id,
            subset=0,
            path="hand.png",
            labeled=True,
            original_name="hand.png",
            class_id=0,
            annotation={"data": {"landmarks": landmarks, "label": 0}},
        )
        session.add(item)
        session.commit()
        item_id = item.id

    response = api.post(
        f"/api/projects/{project_id}/data_items/{item_id}/class_id",
        json={"class_id": 3},
    )
    assert response.status_code == 200, response.text

    with Session(db_manager.get_project_engine(project_id)) as session:
        item = session.query(DataItem).filter(DataItem.id == item_id).one()
        assert item.class_id == 3
        assert item.annotation == {"data": {"landmarks": landmarks, "label": 3}}


def test_marking_classification_item_unlabelled_clears_labeled_flag(api, project_id):
    uploaded = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("worker.png", png_bytes(), "image/png")},
    )
    assert uploaded.status_code == 200, uploaded.text
    item = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"][0]

    response = api.post(
        f"/api/projects/{project_id}/data_items/{item['id']}/class_id",
        json={"class_id": -1},
    )
    assert response.status_code == 200, response.text

    updated = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"][0]
    assert updated["class_id"] == -1
    assert updated["labeled"] is False


@pytest.mark.parametrize("class_id", [True, "0", -2, 999])
def test_class_update_rejects_invalid_or_unknown_ids(api, project_id, class_id):
    api.patch(
        f"/api/projects/{project_id}",
        json={"labels": json.dumps([{"name": "known", "color": "#fff", "id": 0}])},
    )
    uploaded = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("worker.png", png_bytes(), "image/png")},
    )
    assert uploaded.status_code == 200, uploaded.text
    item = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"][0]

    response = api.post(
        f"/api/projects/{project_id}/data_items/{item['id']}/class_id",
        json={"class_id": class_id},
    )

    assert response.status_code == 422
    unchanged = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"][
        0
    ]
    assert unchanged["class_id"] == -1
    assert unchanged["labeled"] is False


def test_download_missing_item_is_404(api, project_id):
    response = api.get(f"/api/projects/{project_id}/data_items/999/download")
    assert response.status_code == 404


def test_random_test_sample_with_no_data(api, project_id):
    """Used by the 'try the model' panel before any data exists."""
    response = api.get(f"/api/projects/{project_id}/data_items/random_test_sample")
    assert response.status_code in (200, 404)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def test_export_status_before_any_export_is_404(api, project_id):
    response = api.get(f"/api/projects/{project_id}/export_status")
    assert response.status_code == 404


def test_export_rejects_an_unknown_format(api, project_id):
    response = api.post(
        f"/api/projects/{project_id}/export_data",
        json={"export_format": "definitely-not-a-format"},
    )
    assert response.status_code >= 400


def test_keypoint_export_rejects_box_only_yolo_format(api):
    created = api.post(
        "/api/projects",
        json={
            "name": "Pose",
            "type": "Keypoint Detection",
            "description": "",
            "labels": [{"id": 0, "name": "nose", "color": "#ff0000"}],
        },
    )
    assert created.status_code == 200, created.text

    response = api.post(
        f"/api/projects/{created.json()['id']}/export_data",
        json={"format": "yolo"},
    )
    assert response.status_code == 400
    assert "YOLO pose" in response.json()["detail"]


def test_download_export_before_exporting_is_404(api, project_id):
    response = api.get(f"/api/projects/{project_id}/download_export")
    assert response.status_code == 404


def test_cleanup_export_when_nothing_was_exported(api, project_id):
    """Cleanup must be idempotent -- the UI calls it defensively."""
    response = api.delete(f"/api/projects/{project_id}/cleanup_export")
    assert response.status_code < 500


def test_export_status_route_is_distinct_from_the_project_one(api, project_id):
    """Regression: these two collided in the generated OpenAPI operation ids.

    They are genuinely different URLs -- '/export_status' here versus
    '/export/status' in the project router -- and both must stay routable.
    """
    dataset_route = api.get(f"/api/projects/{project_id}/export_status")
    project_route = api.get(f"/api/projects/{project_id}/export/status")
    assert dataset_route.status_code == 404
    assert project_route.status_code == 404


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------


def test_delete_all_data_items_on_an_empty_project(api, project_id):
    response = api.request(
        "DELETE", f"/api/projects/{project_id}/data_items", json={"item_ids": []}
    )
    assert response.status_code < 500


def test_class_distribution_survives_a_handpose_annotation(api):
    """Not every annotation is a list of shapes.

    Handpose stores a dict of hand landmarks under the same `data` key, and its
    class is in class_id. Iterating it as shapes yields dictionary *keys* --
    strings -- and the endpoint died with a 500 on any real handpose project.
    """
    created = api.post(
        "/api/projects",
        json={"name": "Hands", "type": "Handpose Classification", "description": ""},
    )
    project_id = created.json()["id"]
    api.patch(
        f"/api/projects/{project_id}",
        json={"labels": '[{"name": "open", "color": "#fff", "id": 0}]'},
    )

    from sqlalchemy.orm import Session

    from anylearning.database import DataItem, Dataset, db_manager

    with Session(db_manager.get_project_engine(project_id)) as session:
        dataset = Dataset()
        session.add(dataset)
        session.flush()
        session.add(
            DataItem(
                dataset_id=dataset.id,
                subset=0,
                path="hand.png",
                labeled=True,
                original_name="hand.png",
                class_id=0,
                annotation={
                    "data": {"landmarks": {"0": {"x": 0.5, "y": 0.5, "z": 0.0}}}
                },
            )
        )
        session.commit()

    body = api.get(f"/api/projects/{project_id}/class_distribution").json()
    assert body["classes"][0]["name"] == "open"
    assert body["classes"][0]["train"] == 1


def test_copy_subset_duplicates_the_images_rather_than_sharing_them(api, project_id):
    """Two rows pointing at one file means deleting either deletes the image."""
    api.post(
        f"/api/projects/{project_id}/upload_data",
        params={"subset": 1},
        files=[
            ("file", ("a.png", png_bytes(), "image/png")),
            ("file", ("b.png", png_bytes(), "image/png")),
        ],
    )

    response = api.post(
        f"/api/projects/{project_id}/data_items/copy_subset",
        json={"from_subset": 1, "to_subset": 2},
    )
    assert response.status_code == 200, response.text
    assert response.json()["copied"] == 2

    test_items = api.get(f"/api/projects/{project_id}/data_items?subset=2").json()
    validation = api.get(f"/api/projects/{project_id}/data_items?subset=1").json()
    assert test_items["total_count"] == 2
    assert validation["total_count"] == 2

    files = list((api.projects_root / str(project_id) / "data").iterdir())
    assert len(files) == 4, "each copy needs its own file on disk"


def test_copy_subset_rejects_a_pointless_copy(api, project_id):
    response = api.post(
        f"/api/projects/{project_id}/data_items/copy_subset",
        json={"from_subset": 1, "to_subset": 1},
    )
    assert response.status_code == 400


def test_copy_subset_of_an_empty_subset_says_so(api, project_id):
    response = api.post(
        f"/api/projects/{project_id}/data_items/copy_subset",
        json={"from_subset": 1, "to_subset": 2},
    )
    assert response.status_code == 400
    assert "no images" in response.json()["detail"]


def _coco_zip(label="cat"):
    """An archive shaped like a COCO export: images plus one annotations file."""
    import json
    import zipfile

    coco = {
        "images": [
            {"id": 1, "file_name": "images/photo.png", "width": 16, "height": 16}
        ],
        "categories": [{"id": 4, "name": label}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 4, "bbox": [2, 3, 8, 6]}
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/photo.png", png_bytes().getvalue())
        archive.writestr("annotations/instances_default.json", json.dumps(coco))
    buffer.seek(0)
    return buffer


def _yolo_zip(label="helmet"):
    """The usual YOLO layout: images/, labels/, and a class list."""
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("images/shot.png", png_bytes().getvalue())
        archive.writestr("labels/shot.txt", "0 0.5 0.5 0.5 0.25\n")
        archive.writestr("classes.txt", f"{label}\n")
    buffer.seek(0)
    return buffer


def test_upload_reads_a_coco_export(api):
    """Export produces COCO; import has to accept it, or the round trip is a
    one-way trip."""
    created = api.post(
        "/api/projects",
        json={"name": "Coco", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]

    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("coco.zip", _coco_zip(), "application/zip")},
    )
    assert response.status_code == 200, response.text

    labels = api.get(f"/api/projects/{project_id}").json()["labels"]
    assert [label["name"] for label in labels] == ["cat"]

    items = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"]
    assert len(items) == 1 and items[0]["labeled"] is True

    annotation = api.get(
        f"/api/projects/{project_id}/data_items/{items[0]['id']}/get_annotation"
    ).json()
    assert annotation[0]["categories"] == ["cat"]
    # bbox [x, y, w, h] -> the four corners, in image pixels.
    assert annotation[0]["points"] == [[2, 3], [10, 3], [10, 9], [2, 9]]


def test_upload_reads_a_yolo_export(api):
    created = api.post(
        "/api/projects",
        json={"name": "Yolo", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]

    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("yolo.zip", _yolo_zip(), "application/zip")},
    )
    assert response.status_code == 200, response.text

    labels = api.get(f"/api/projects/{project_id}").json()["labels"]
    assert [label["name"] for label in labels] == ["helmet"]

    items = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"]
    annotation = api.get(
        f"/api/projects/{project_id}/data_items/{items[0]['id']}/get_annotation"
    ).json()
    # Normalised centre/size against a 16x16 image: half width, quarter height,
    # centred. Without the image size every box would land in the corner.
    assert annotation[0]["points"] == [[4, 6], [12, 6], [12, 10], [4, 10]]


def test_yolo_labels_without_class_names_are_left_alone(api):
    """A .txt sidecar carries no format marker and no class names. Guessing
    would invent labels; the image is imported unlabelled instead."""
    import zipfile

    created = api.post(
        "/api/projects",
        json={"name": "Nameless", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("shot.png", png_bytes().getvalue())
        archive.writestr("shot.txt", "0 0.5 0.5 0.5 0.25\n")
    buffer.seek(0)

    api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("yolo.zip", buffer, "application/zip")},
    )
    items = api.get(f"/api/projects/{project_id}/data_items").json()["data_items"]
    assert len(items) == 1
    assert items[0]["labeled"] is False
    assert not api.get(f"/api/projects/{project_id}").json()["labels"]


def test_label_ids_do_not_collide_after_a_deletion(api):
    """`len(labels)` as the next id is wrong the moment one has been deleted.

    Classification stores `class_id` pointing at a label id, so a duplicate id
    silently reads every image of one class as the other -- a data bug that
    looks like a mislabelled dataset.
    """
    created = api.post(
        "/api/projects",
        json={"name": "Ids", "type": "Object Detection", "description": ""},
    )
    project_id = created.json()["id"]

    api.patch(
        f"/api/projects/{project_id}",
        json={
            "labels": json.dumps(
                [
                    {"name": "kept", "color": "#111111", "id": 0},
                    {"name": "also-kept", "color": "#222222", "id": 2},
                ]
            )
        },
    )

    api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("d.zip", _zip_with_polygon_labels("new"), "application/zip")},
    )

    labels = api.get(f"/api/projects/{project_id}").json()["labels"]
    ids = [label["id"] for label in labels]
    assert len(ids) == len(set(ids)), f"duplicate label ids: {labels}"
    assert [label["name"] for label in labels] == ["kept", "also-kept", "new"]
    assert max(ids) == 3


def test_the_upload_reports_which_classes_it_created(api):
    """ "It did not pick up my categories" should be visible, not inferred."""
    created = api.post(
        "/api/projects",
        json={"name": "Reported", "type": "Image Segmentation", "description": ""},
    )
    project_id = created.json()["id"]

    api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("d.zip", _zip_with_polygon_labels("cell"), "application/zip")},
    )

    status = api.get(f"/api/projects/{project_id}/upload_status").json()
    assert status["created_labels"] == ["cell"]


def test_an_upload_with_no_annotations_reports_no_classes(api, project_id):
    """The useful distinction: the archive had none, rather than the app having
    ignored them."""
    api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("plain.zip", _zip_with_images(), "application/zip")},
    )
    status = api.get(f"/api/projects/{project_id}/upload_status").json()
    assert status["created_labels"] == []


# --------------------------------------------------------------------------
# Trying a model on an image the project already has
# --------------------------------------------------------------------------


def test_a_random_test_sample_falls_back_to_validation(api, project_id):
    """A project with an empty test split is ordinary, not an error.

    "Use a test image" answered "Failed to fetch test sample" for every project
    that had never filled its third split -- which was both of the real ones on
    this machine -- when an image the model can be tried on was sitting one
    split over. The response says which split it came from so the dialog can be
    honest about it.
    """
    response = api.post(
        f"/api/projects/{project_id}/upload_data",
        files={"file": ("val.png", png_bytes(), "image/png")},
        params={"subset": 1},
    )
    assert response.status_code == 200, response.text

    sample = api.get(f"/api/projects/{project_id}/data_items/random_test_sample")
    assert sample.status_code == 200, sample.text
    assert sample.headers["x-anylearning-subset"] == "validation"
    assert len(sample.content) > 0


def test_a_random_test_sample_prefers_the_test_split(api, project_id):
    for subset in (1, 2):
        response = api.post(
            f"/api/projects/{project_id}/upload_data",
            files={"file": (f"image_{subset}.png", png_bytes(), "image/png")},
            params={"subset": subset},
        )
        assert response.status_code == 200, response.text

    sample = api.get(f"/api/projects/{project_id}/data_items/random_test_sample")
    assert sample.status_code == 200, sample.text
    assert sample.headers["x-anylearning-subset"] == "test"


def test_a_project_with_no_images_says_what_to_do(api, project_id):
    sample = api.get(f"/api/projects/{project_id}/data_items/random_test_sample")
    assert sample.status_code == 404
    detail = sample.json()["detail"]
    assert "test or validation" in detail
    assert "upload" in detail.lower()

import json
import yaml
import numpy as np
import supervisely_lib as sly


def visualize(img: np.ndarray, ann: sly.Annotation, name, roi: sly.Rectangle = None):
    vis = img.copy()
    if roi is not None:
        roi.draw_contour(vis, color=[255, 0, 0], thickness=3)
    ann.draw_contour(vis, thickness=3)
    sly.image.write(f"./images/{name}", vis)


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 2723

    # get information about model
    info = api.task.send_request(task_id, "get_session_info", data={})
    print("Information about deployed model:")
    print(json.dumps(info, indent=4))

    # get model output classes and tags
    meta_json = api.task.send_request(task_id, "get_output_classes_and_tags", data={})
    model_meta = sly.ProjectMeta.from_json(meta_json)
    print("Model produces following classes and tags")
    print(model_meta)

    # get model inference settings (optional)
    resp = api.task.send_request(task_id, "get_custom_inference_settings", data={})
    settings_yaml = resp["settings"]
    settings = yaml.safe_load(settings_yaml)
    # you can change this default settings and pass them to any inference method
    print("Model inference settings:")
    print(json.dumps(settings, indent=4))

    # inference for url
    image_url = "https://i.imgur.com/tEkCb69.jpg"

    # download image for further debug visualizations
    save_path = f"./images/{sly.fs.get_file_name_with_ext(image_url)}"
    sly.fs.ensure_base_path(save_path)  # create directories if needed
    sly.fs.download(image_url, save_path)
    img = sly.image.read(save_path)  # RGB

    # apply model to image URl (full image)
    # you can pass 'settings' dictionary to any inference method
    # every model defines custom inference settings
    ann_json = api.task.send_request(task_id, "inference_image_url",
                                     data={
                                         "image_url": image_url,
                                         "settings": settings,
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "01_prediction_url.jpg")

    # apply model to image URL (only ROI - region of interest)
    height, width = img.shape[0], img.shape[1]
    top, left, bottom, right = 0, 0, height - 1, int(width/2)
    roi = sly.Rectangle(top, left, bottom, right)
    ann_json = api.task.send_request(task_id, "inference_image_url",
                                     data={
                                         "image_url": image_url,
                                         "rectangle": [top, left, bottom, right]
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    visualize(img, ann, "02_prediction_url_roi.jpg", roi)

    # apply model to image id (full image)
    image_id = 770730
    ann_json = api.task.send_request(task_id, "inference_image_id", data={"image_id": image_id})
    ann = sly.Annotation.from_json(ann_json, model_meta)
    img = api.image.download_np(image_id)
    sly.image.write("./images/03_input_id.jpg", img)
    visualize(img, ann, "03_prediction_id.jpg")

    # apply model to image id (only ROI - region of interest)
    image_id = 770730
    img = api.image.download_np(image_id)
    height, width = img.shape[0], img.shape[1]
    top, left, bottom, right = 0, 0, height - 1, int(width / 2)
    roi = sly.Rectangle(top, left, bottom, right)
    ann_json = api.task.send_request(task_id, "inference_image_id",
                                     data={
                                         "image_id": image_id,
                                         "rectangle": [top, left, bottom, right]
                                     })
    ann = sly.Annotation.from_json(ann_json, model_meta)
    sly.image.write("./images/04_input_id_roi.jpg", img)
    visualize(img, ann, "04_prediction_id_roi.jpg", roi)

    # apply model to several images (using id)
    batch_ids = [770730, 770727, 770729, 770720]
    resp = api.task.send_request(task_id, "inference_batch_ids", data={"batch_ids": batch_ids})
    for ind, (image_id, ann_json) in enumerate(zip(batch_ids, resp)):
        ann = sly.Annotation.from_json(ann_json, model_meta)
        img = api.image.download_np(image_id)
        sly.image.write(f"./images/05_input_batch_{ind:03d}_{image_id}.jpg", img)
        visualize(img, ann, f"05_prediction_batch_{ind:03d}_{image_id}.jpg")


if __name__ == "__main__":
    main()

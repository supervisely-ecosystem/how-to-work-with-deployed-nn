import json
import supervisely_lib as sly


def visualize(img, ann, name):
    vis = img.copy()
    ann.draw_contour(vis, thickness=3)
    sly.image.write(f"./images/{name}", vis)


def main():
    api = sly.Api.from_env()

    # task id of the deployed model
    task_id = 2719

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
    settings = api.task.send_request(task_id, "get_custom_inference_settings", data={})
    print("Model inference settings:")
    print(json.dumps(settings, indent=4))

    # inference for url
    image_url = "https://i.imgur.com/tEkCb69.jpg"

    # download image for further debug visualizations
    save_path = f"./images/{sly.fs.get_file_name_with_ext(image_url)}"
    sly.fs.ensure_base_path(save_path)  # create directories if needed
    sly.fs.download(image_url, save_path)
    img = sly.image.read(save_path)  # RGB

    # apply model to image URl
    ann_json = api.task.send_request(task_id, "inference_image_url", data={"image_url": image_url})
    ann = sly.Annotation.from_json(ann_json, model_meta)

    # visualize prediction
    visualize(img, ann, "01_prediction_full_image.jpg")









if __name__ == "__main__":
    main()
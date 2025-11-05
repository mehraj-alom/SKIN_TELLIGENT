from src.inference.pipeline import DetectionAndClassificationPipeline
from logger import logger

def main():
    config_path = "src/config/pipeline_config.yaml"



    pipeline = DetectionAndClassificationPipeline(config_path=config_path)
    output_img, crops , classification_result  = pipeline.run()

    logger.info(f"Pipeline finished. {len(crops)} regions saved.")

if __name__ == "__main__":
    main()

from src.inference.pipeline import DetectionPipeline
from logger import logger

def main():
    config_path = "src/config/pipeline_config.yaml"



    pipeline = DetectionPipeline(config_path=config_path)
    output_img, crops = pipeline.run()

    logger.info(f"Pipeline finished. {len(crops)} regions saved.")

if __name__ == "__main__":
    main()

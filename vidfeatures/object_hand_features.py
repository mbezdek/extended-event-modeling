from utils import logger, parse_config, gen_feature_video

if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')

    gen_feature_video(track_csv=args.input_track_csv, skel_csv=args.input_skel_csv,
                      output_csv=args.output_objhand_csv)

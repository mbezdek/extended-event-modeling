import pandas as pd
from utils import logger, parse_config,\
    calc_joint_dist, calc_joint_speed, calc_joint_acceleration,\
    calc_interhand_dist, calc_interhand_speed, calc_interhand_acceleration

if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')

    # Load skeleton dataframe
    skeldf = pd.read_csv(args.skel_csv_in)
    if args.joints == "all":
        joints = list(range(25))
    for j in joints:
        skeldf = calc_joint_dist(skeldf,j)
        skeldf = calc_joint_speed(skeldf, j)
        skeldf = calc_joint_acceleration(skeldf,j)
    skeldf = calc_interhand_dist(skeldf)
    skeldf = calc_interhand_speed(skeldf)
    skeldf = calc_interhand_acceleration(skeldf)
    skeldf.to_csv(args.skel_csv_out)
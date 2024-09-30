import numpy as np
import pandas as pd
import pickle as pkl
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
# from src.utils import parse_config, logger, DictObj


class PCAComputer:
    """
    This class loads individual features and compute PCA matrices for these features, based on all activities.
    """

    def __init__(self, configs):
        self.feature_tag = configs.feature_tag
        self.sample = int(configs.n_sample)
        self.pca_tag = configs.pca_tag
        self.appear_components = int(configs.appear_components)
        self.optical_components = int(configs.optical_components)
        self.skel_components = int(configs.skel_components)
        self.emb_components = int(configs.emb_components)
        self.n_components = self.appear_components + self.optical_components + \
                            self.skel_components + self.emb_components
        runs = open(f"{args.run}", 'r').readlines()
        runs = [x.strip() for x in runs]
        self.runs = runs

    def load_and_sample(self, path: str) -> pd.DataFrame:
        """
        :param path: path to resampled features that was constructed by FeatureProcessor
        :return:
        """
        df_dict = pkl.load(open(path, 'rb'))
        input_df = DictObj(df_dict)
        try:
            res = input_df.combined_resampled_df.sample(n=self.sample)
            return res
        except Exception as e:
            print(f'Failed: Path={path}, len={len(input_df.combined_resampled_df)}')
            return pd.DataFrame()

    def compute_and_save_pca_components(self) -> None:
        """
        compute PCA matrices for all (4) types of features
        :return:
        """
        print(f'Feature Tag is: {self.feature_tag}')
        print(f'Sample for each run is: {self.sample}')
        print(f'PCA tag is : {self.pca_tag}')
        print(f'# components: {self.n_components}')

        # load all combined_resampled_df dataframes and resample
        input_paths = [f'output/preprocessed_features/{run}_{self.feature_tag}.pkl' for run in self.runs]
        print(f'Total runs: {len(input_paths)}')

        input_dfs = Parallel(n_jobs=16)(delayed(self.load_and_sample)(path) for path in input_paths)
        combined_runs = pd.concat(input_dfs, axis=0)
        print(f'Total data points to PCA: {len(combined_runs)}')
        # run pca and save pca pickles
        pca = PCA(n_components=self.n_components, whiten=True)
        pca.fit(combined_runs)
        print(f"pca.components_.shape={pca.components_.shape}")
        print(f'Saving {self.feature_tag}_{self.pca_tag}_pca.pkl')
        pkl.dump(pca, open(f'output/{self.feature_tag}_{self.pca_tag}_pca.pkl', 'wb'))

        pca = PCA(n_components=self.appear_components, whiten=True)
        pca.fit(combined_runs.iloc[:, :2])
        print(f"pca.components_.shape={pca.components_.shape}")
        print(f'Saving {self.feature_tag}_{self.pca_tag}_appear_pca.pkl')
        pkl.dump(pca, open(f'output/{self.feature_tag}_{self.pca_tag}_appear_pca.pkl', 'wb'))

        pca = PCA(n_components=self.optical_components, whiten=True)
        pca.fit(combined_runs.iloc[:, 2:4])
        print(f"pca.components_.shape={pca.components_.shape}")
        print(f'Saving {self.feature_tag}_{self.pca_tag}_optical_pca.pkl')
        pkl.dump(pca, open(f'output/{self.feature_tag}_{self.pca_tag}_optical_pca.pkl', 'wb'))

        pca = PCA(n_components=self.skel_components, whiten=True)
        pca.fit(combined_runs.iloc[:, 4:-100])
        print(f"pca.components_.shape={pca.components_.shape}")
        print(f'Saving {self.feature_tag}_{self.pca_tag}_skel_pca.pkl')
        pkl.dump(pca, open(f'output/{self.feature_tag}_{self.pca_tag}_skel_pca.pkl', 'wb'))

        pca = PCA(n_components=self.emb_components, whiten=True)
        pca.fit(combined_runs.iloc[:, -100:])
        print(f"pca.components_.shape={pca.components_.shape}")
        print(f'Saving {self.feature_tag}_{self.pca_tag}_emb_pca.pkl')
        pkl.dump(pca, open(f'output/{self.feature_tag}_{self.pca_tag}_emb_pca.pkl', 'wb'))
        print('Done!')


class PCATransformer:
    """
    This class loads individual PCA matrices and perform transform or invert based on these PCA matrices
    PCA matrices were derived by PCAComputer class
    """

    def __init__(self, feature_tag, pca_tag):
        self.feature_tag = feature_tag
        self.pca_tag = pca_tag
        # self.pca_all = self.load_pca_version_agnostic(f'output/{self.feature_tag}_{self.pca_tag}_pca')
        self.pca_appear = PCATransformer.load_pca_version_agnostic(f'output/pca_estimator_from_all_runs/{self.feature_tag}_{self.pca_tag}_appear_pca')
        self.pca_optical = PCATransformer.load_pca_version_agnostic(f'output/pca_estimator_from_all_runs/{self.feature_tag}_{self.pca_tag}_optical_pca')
        self.pca_skel = PCATransformer.load_pca_version_agnostic(f'output/pca_estimator_from_all_runs/{self.feature_tag}_{self.pca_tag}_skel_pca')
        self.pca_emb = PCATransformer.load_pca_version_agnostic(f'output/pca_estimator_from_all_runs/{self.feature_tag}_{self.pca_tag}_emb_pca')

    @staticmethod
    def load_pca_version_agnostic(input_prefix):
        # Load components
        components = np.load(f"{input_prefix}_components.npy")
        
        # Load mean
        mean = np.load(f"{input_prefix}_mean.npy")
        
        # Load explained variance ratio
        explained_variance_ratio = np.load(f"{input_prefix}_explained_variance_ratio.npy")
        
        # Load explained variance
        explained_variance = np.load(f"{input_prefix}_explained_variance.npy")
        
        # Load singular values
        singular_values = np.load(f"{input_prefix}_singular_values.npy")
        
        # Load n_components
        n_components = np.load(f"{input_prefix}_n_components.npy")[0]
        
        # Load whiten flag
        whiten = bool(np.load(f"{input_prefix}_whiten.npy")[0])

        # Load n_features_in
        n_features_in = np.load(f"{input_prefix}_n_features_in.npy")[0]

        # Load noise_variance
        noise_variance = np.load(f"{input_prefix}_noise_variance.npy")

        # Load n_samples
        n_samples = np.load(f"{input_prefix}_n_samples.npy")[0]
        
        # Create a new PCA object
        pca = PCA(n_components=n_components, whiten=whiten)
        
        # Set the attributes
        pca.n_features_in_ = n_features_in
        pca.noise_variance_ = noise_variance
        pca.n_samples_ = n_samples
        pca.components_ = components
        pca.mean_ = mean
        pca.explained_variance_ratio_ = explained_variance_ratio
        pca.explained_variance_ = explained_variance
        pca.singular_values_ = singular_values
        pca.n_components_ = n_components
        
        return pca

    def transform(self, original_feature_array: np.ndarray) -> np.ndarray:
        pca_feature_array_appear = self.pca_appear.transform(original_feature_array[:, :2])
        pca_feature_array_optical = self.pca_optical.transform(original_feature_array[:, 2:4])
        pca_feature_array_skel = self.pca_skel.transform(original_feature_array[:, 4:-100])
        pca_feature_array_emb = self.pca_emb.transform(original_feature_array[:, -100:])
        pca_feature_array = np.hstack([pca_feature_array_appear, pca_feature_array_optical,
                                       pca_feature_array_skel, pca_feature_array_emb])
        return pca_feature_array

    def invert_transform(self, pca_feature_array: np.ndarray) -> np.ndarray:
        indices = [self.pca_appear.n_components,
                   self.pca_appear.n_components + self.pca_optical.n_components,
                   self.pca_appear.n_components + self.pca_optical.n_components + self.pca_skel.n_components,
                   self.pca_appear.n_components + self.pca_optical.n_components +
                   self.pca_skel.n_components + self.pca_emb.n_components]
        original_inverted_array_appear = self.pca_appear.inverse_transform(pca_feature_array[:, :indices[0]])
        original_inverted_array_optical = self.pca_optical.inverse_transform(pca_feature_array[:, indices[0]:indices[1]])
        original_inverted_array_skel = self.pca_skel.inverse_transform(pca_feature_array[:, indices[1]:indices[2]])
        original_inverted_array_emb = self.pca_emb.inverse_transform(pca_feature_array[:, indices[2]:])
        original_inverted_array = np.hstack(
            [original_inverted_array_appear, original_inverted_array_optical,
             original_inverted_array_skel, original_inverted_array_emb])

        return original_inverted_array


if __name__ == "__main__":
    args = parse_config()
    logger.info(f'Config: {args}')
    pca_processor = PCAComputer(configs=args)
    pca_processor.compute_and_save_pca_components()

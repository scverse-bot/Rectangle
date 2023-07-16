import numpy as np
import pandas as pd
import src.rectanglepy as rectangle
from src.rectanglepy.pp.create_signature import convert_to_cpm, create_bias_factors, get_marker_genes


class ParameterOptimizer:
    def __init__(self, sc_data: pd.DataFrame, annotations: pd.Series, pseudobulk_sig: pd.DataFrame, de_results):
        self.sc_data = sc_data[sc_data.sum(axis=1) > 0]
        self.annotations = annotations
        self.pseudobulk_sig = pseudobulk_sig
        self.de_results = de_results

    def optimize_parameters(self):
        """Optimize the parameters of the model. via grid search.

        Returns
        -------
        optimized_p: float
            The optimized value of p.
        optimized lfc: float
            The optimized value of lfc.
        """
        logfcs = [x / 100 for x in range(40, 380, 20)]  # 0.4 to 3.6
        ps = [x / 1000 for x in range(1, 52, 3)]  # 0.001 to 0.049

        best_p, best_lfc, best_rmse = None, None, None

        for p in ps:
            for logfc in logfcs:
                print("Computing RMSE for p: ", p, " and lfc: ", logfc, "...")
                rmse = self.calculate_deconvolution(logfc, p)
                if best_rmse is None or rmse < best_rmse:
                    best_rmse = rmse
                    best_p = p
                    best_lfc = logfc

        print("Best RMSE: ", best_rmse)
        print("Best p: ", best_p)
        print("Best lfc: ", best_lfc)
        return best_p, best_lfc

    def calculate_deconvolution(self, starting_lfc, starting_p):
        number_of_bulks = 20
        bulks, real_fractions = self.generate_pseudo_bulks(number_of_bulks)
        estimated_fractions = self.generate_estimated_fractions(bulks, starting_p, starting_lfc)
        # sort by index
        real_fractions = real_fractions.sort_index()
        estimated_fractions = estimated_fractions.sort_index()

        rsme = self.calculate_rsme(real_fractions, estimated_fractions)
        return rsme

    def generate_estimated_fractions(self, bulks, p, logfc):
        signature = self.generate_signature(logfc, p)
        bias_factors = create_bias_factors(self.pseudobulk_sig)
        signature = signature * bias_factors
        estimated_fractions = bulks.apply(lambda x: self.QP(signature, x), axis=0)
        # remove "unknown" cell type
        # estimated_fractions = estimated_fractions.drop("Unknown", axis=0)
        estimated_fractions.index = signature.columns
        return estimated_fractions

    def QP(self, signature, bulk):
        genes = list(set(signature.index) & set(bulk.index))
        signature = signature.loc[genes].sort_index()
        bulk = bulk.loc[genes].sort_index().astype("double")
        return rectangle.tl.solve_dampened_wsl(signature, bulk)

    def generate_signature(self, logfc, p):
        marker_genes = pd.Series(get_marker_genes(self.annotations, self.de_results, logfc, p, self.sc_data))

        signature = convert_to_cpm(self.pseudobulk_sig).loc[marker_genes]
        return signature

    def generate_pseudo_bulks(self, number_of_bulks):
        annotation_counts = self.annotations.value_counts()

        split_size = (len(self.annotations) / len(annotation_counts)) / 5
        max_allowed_size = annotation_counts.min() if annotation_counts.min() > split_size else split_size
        bulks = []
        real_fractions = []
        for _ in range(number_of_bulks):
            indices = []
            cell_numbers = []
            for annotation in self.annotations.unique():
                annotation_indices = self.annotations[self.annotations == annotation].index
                upper_limit = min(max_allowed_size, len(annotation_indices))
                number_of_cells = np.random.randint(0, upper_limit)
                cell_numbers.append(number_of_cells)
                random_annotation_indices = np.random.choice(annotation_indices, number_of_cells, replace=False)
                indices.extend(random_annotation_indices)

            random_cells = self.sc_data.loc[:, indices]
            random_cells_sum = random_cells.sum(axis=1)
            pseudo_bulk = random_cells_sum * 1e6 / np.sum(random_cells_sum)
            bulks.append(pseudo_bulk)

            cell_fractions = np.array(cell_numbers) / np.sum(cell_numbers)
            cell_fractions = pd.Series(cell_fractions, index=self.annotations.unique())
            real_fractions.append(cell_fractions)
        return pd.DataFrame(bulks).T, pd.DataFrame(real_fractions).T

    def calculate_rsme(self, real_fractions: pd.Series, predicted_fractions: pd.Series):
        return np.sqrt(np.mean((real_fractions - predicted_fractions) ** 2))

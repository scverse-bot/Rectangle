import numpy as np
import pandas as pd
import src.rectanglepy as rectangle
from scipy.stats import pearsonr
from src.rectanglepy.pp.create_signature import convert_to_cpm, create_bias_factors, get_marker_genes


class ParameterOptimizer:
    def __init__(self, sc_data: pd.DataFrame, annotations: pd.Series, pseudobulk_sig: pd.DataFrame, de_results):
        self.sc_data = sc_data[sc_data.sum(axis=1) > 0]
        self.annotations = annotations
        self.pseudobulk_sig = pseudobulk_sig[pseudobulk_sig.sum(axis=1) > 0]
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
        logfcs = [x / 100 for x in range(80, 130, 10)]
        ps = [x / 1000 for x in range(18, 24, 1)]
        best_p, best_lfc, best_rmse = None, None, None
        results = []
        for p in ps:
            for logfc in logfcs:
                print("Computing RMSE and correlation for p: ", p, " and lfc: ", logfc, "...")
                rmse, pearson_r, rmse_dict = self.calculate_deconvolution(logfc, p)
                results.append({"p": p, "lfc": logfc, "rmse": rmse, "pearson_r": pearson_r, "rmse_dict": rmse_dict})
                if best_rmse is None or rmse < best_rmse:
                    best_rmse = rmse
                    best_p = p
                    best_lfc = logfc
        results_df = pd.DataFrame(results)
        print("Best Pearson R: ", best_rmse)
        print("Best RMSE: ", best_rmse)
        print("Best p: ", best_p)
        print("Best lfc: ", best_lfc)
        print(results_df)
        results_df.to_csv("./results_opt_3.csv")
        return best_p, best_lfc

    def calculate_deconvolution(self, starting_lfc, starting_p):
        number_of_bulks = 30
        bulks, real_fractions = self.generate_pseudo_bulks(number_of_bulks)
        estimated_fractions = self.generate_estimated_fractions(bulks, starting_p, starting_lfc)

        real_fractions = real_fractions.sort_index()
        estimated_fractions = estimated_fractions.sort_index()

        rsme, rsme_dict = self.calculate_rsme(real_fractions, estimated_fractions)
        pearson_r = self.calculate_correlation(real_fractions, estimated_fractions)
        return rsme, pearson_r, rsme_dict

    def generate_estimated_fractions(self, bulks, p, logfc):
        signature = self.generate_signature(logfc, p)
        pseudo_sig = convert_to_cpm(self.pseudobulk_sig)
        bias_factors = create_bias_factors(self.pseudobulk_sig)
        signature = signature * bias_factors
        estimated_fractions = bulks.apply(lambda x: self.QP(signature, x), axis=0)
        estimated_fractions.index = signature.columns

        estimated_fractions_corrected = []
        for i in range(len(estimated_fractions.columns)):
            estimated_fractions_corrected.append(
                rectangle.tl.correct_for_unknown_cell_content(
                    bulks.iloc[:, i], pseudo_sig, estimated_fractions.iloc[:, i], bias_factors
                )
            )

        # remove "unknown" cell type
        # estimated_fractions = estimated_fractions.drop("Unknown", axis=0)
        estimated_fractions_corrected = pd.DataFrame(estimated_fractions_corrected).T
        estimated_fractions_corrected.drop("Unknown", axis=0, inplace=True)
        return estimated_fractions_corrected

    def QP(self, signature, bulk):
        genes = list(set(signature.index) & set(bulk.index))
        signature = signature.loc[genes].sort_index()
        bulk = bulk.loc[genes].sort_index().astype("double")
        return rectangle.tl.solve_dampened_wsl(signature, bulk)

    def generate_signature(self, logfc, p):
        marker_genes = pd.Series(get_marker_genes(self.annotations, self.de_results, logfc, p, self.sc_data))
        print("Number of marker genes: ", len(marker_genes))
        signature = convert_to_cpm(self.pseudobulk_sig).loc[marker_genes]
        return signature

    def generate_pseudo_bulks(self, number_of_bulks):
        # we only take relatively small number of cells per annotation, to have higher variance
        split_size = 60
        bulks = []
        real_fractions = []
        for _ in range(number_of_bulks):
            indices = []
            cell_numbers = []
            for annotation in self.annotations.unique():
                annotation_indices = self.annotations[self.annotations == annotation].index
                upper_limit = min(split_size, len(annotation_indices))
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

    def calculate_rsme(self, real_fractions: pd.DataFrame, predicted_fractions: pd.DataFrame):
        rmse_dict = {}

        for index, (row1, row2) in enumerate(zip(real_fractions.values, predicted_fractions.values)):
            rmse = np.sqrt(((row1 - row2) ** 2).mean())
            rmse_dict[real_fractions.index[index]] = rmse
        total_rmse = np.sqrt(np.mean((real_fractions - predicted_fractions) ** 2))
        return total_rmse, rmse_dict

    def calculate_correlation(self, real_fractions: pd.DataFrame, predicted_fractions: pd.DataFrame):
        return pearsonr(real_fractions.values.flatten(), predicted_fractions.values.flatten())[0]

from selection import Selector
from marginal_fitter import MarginalFitter

def asset_names_from_selected_pairs(selected_pairs):
    asset_names = set()
    for c1, c2, _ in selected_pairs:
        asset_names.add(c1)
        asset_names.add(c2)
    asset_names = list(asset_names)
    return asset_names

if __name__ == "__main__":
    selector = Selector()
    selected_pairs = selector.run(interval='1h', days_back=10)
    asset_names = asset_names_from_selected_pairs(selected_pairs)

    fitter = MarginalFitter()

    summary = fitter.fit_assets(selector.data, asset_names)
    for asset, result in summary.items():
        print(f"Best fit for {asset}: {result['best_dist']}")
        print(result['uniform'])
### Next

- Add support for custom encoder models. See https://github.com/flekschas/peax-avocado for an example
- Add support for custom data directory via the `--base-data-dir` argument of `start.py` 

### v0.4.0

This is the version described in our paper (Lekschas et al., 2020).

- Add support for changing the classifier via the config file's `classifier` and `classifier_params` properties
- Add ability to show more than 5 results per page via to HiGlass' new view scrolling
- Add legend to the UMAP projection view and highlight
- Add legend for the x-axis to the progress views
- Add explanatory help for the progress views
- Add ability to sort selected window by prediction probability
- Add ability to normalize tracks within the same window by setting `normalize_tracks: true` in the config file. This is useful for exploring differential peaks.
- Add ability to show the window with the highest prediction probability in the query view instead of a fixed search window by setting `variable_target: true` in the config file. This is useful for exploring pre-loaded labels where there is no defined search query
- Improve the visibility of the _Re-Train_ and _Compute Projection_ buttons
- Update HiGlass to `v1.7`
- Fix several minor bugs

### v0.3.0

- Support multitrack search
- Update active learning sampling strategies
- Make windows selectable
- Visualize the progress of the actively-learned classifier
- Update and expand examples to use our newly trained autoencoders
- Tons of bug fixes and performance improvements

### v0.2.0

- Represent datasets and encoders as classes rather than loose collections of `dict`s
- Make autoencoders optional, i.e., some tracks might be encodable (and thus searchable) but the encoding might not be decodable.
- Cache chunked, encoded, and autoencoded data as HDF5 files to avoid having to hold everything in memory.
- Replace internal scatterplot with [`regl-scatterplot`](https://github.com/flekschas/regl-scatterplot)
- Update the UI to the latest version of HiGlassApp
- Remove Bootstrap v3, which is used by HiGlass's view header, as it's not needed and imposes a security risk

### v0.1.0

- Search across a single bigWig datasets over up to a few chromosomes

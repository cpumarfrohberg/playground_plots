<h1 align="center">Showcase plots</h1>
<p align="center">Project Description</p>
Showcase selected plots for drawing initial conclusions on toy dataset `mtcars`.

## Content of the project
* 1. `explore_data` directory: implement class for reading data and plotting
* 2. `app.py`
* 3. `requirements.txt`

## For running the analysis in designated environment
* clone repo locally
* run `transform.py`for transforming data and using transformed data for example dashboard
* create an environment with the contents of the `requirements.txt` file; if you are using conda: install pip first (e.g. via `conda install pip` and then `pip install -r requirements.txt`)
* run `streamlit run app.py` for seeing an example dashboard for the analysis.

## For running the app in a docker container
* clone repo locally
* build image with
`docker build -t streamlitchurnapp:latest -f docker/Dockerfile .`
* run image with
`docker run --rm -p 8501:8501 --name test_streamlit_app streamlitchurnapp:latest`
* in your web browser: map your localhost to port 8501 in container

## For running on your browser/yur phone
* hit this url https://firstry.streamlit.app


## Next steps
* [ ] plot correlations
* [ ] plots for categorical data
* [ ] run statistical tests to complement first conclusions based on plots

## Author
**Carlos Pumar-Frohberg**

- [Profile](https://github.com/cpumarfrohberg)
- [Email](mailto:cpumarfrohberg@gmail.com?subject=Hi "Hi!")


## 🤝 Support

Comments, questions and/or feedback are welcome!

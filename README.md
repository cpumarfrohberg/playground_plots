<h1 align="center">Compare new and repeated clients.</h1>
<p align="center">Project Description</p>
Investigate potential difference between new and repeated clients.

## Content of the project
* 1. `explore_data` directory: implement class for data transformation and plotting
* 2. `app.py`
* 3. `requirements.txt`
* 4. `.ipynb`: template notebooks #TODO: update

## For running the analysis in designated environment
* clone repo locally
* run `transform.py`for transforming data and using transformed data for example dashboard
* create an environment with the contents of the requirements.txt file (if you are using conda: install pip first (e.g. via `conda install pip` and then `pip install -r requirements.txt`)
* run `streamlit run app.py` for seeing an example dashboard for the analysis.

## For running the app in a docker container
* clone repo locally
* build image with
`docker build -t streamlitchurnapp:latest -f docker/Dockerfile .`
* run image with
`docker run -p 8501:8501 streamlitchurnapp:latest`
* in your web browser: map your localhost to port 8501 in container

## Next steps
* [ ] 

## Author
**Carlos Pumar-Frohberg**

- [Profile](https://github.com/cpumarfrohberg)
- [Email](mailto:cpumarfrohberg@gmail.com?subject=Hi "Hi!")


## 🤝 Support

Comments, questions and/or feedback are welcome!

# ask_theology

To see the deployed app in action, go to https://asktheology.streamlit.app/.

If the app is inactive (likely the case), click on the option to restart the app and please wait patiently as it boots up.

## Running the app locally

To run the app locally, create a python environment (conda or venv) and run `pip install -r requirements.txt`.

NOTE: This requirements file is only valid for Python versions 3.9-3.11. 3.12 and above, the install for numpy errors out due to changes to the pkgutil package. See this [thread](https://stackoverflow.com/questions/77364550/attributeerror-module-pkgutil-has-no-attribute-impimporter-did-you-mean) for more details.

You will also need to supply API keys for OpenAI and HuggingFace in a separate `secrets.toml` file. The contents of the file should look like this:

```
HF_API_TOKEN = "XXX"
OPENAI_API_TOKEN = "XXX"
```

For more information on setting up secrets management in Streamlit, you can go [here](https://docs.streamlit.io/develop/concepts/connections/secrets-management).

## The dataset

The dataset is comprised of text chunks taken from five academic Christian resources which are converted to embeddings via the [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model.

The dataset is publicly available on HuggingFace [here](https://huggingface.co/datasets/hxyue1/ask_theology).

The books included in the dataset are:

Carson, Donald A, and Douglas J Moo. An Introduction to the New Testament. Grand Rapids, Michigan., Zondervan, 2005.
Cowan, Steven B, et al. Five Views on Apologetics. Grand Rapids, Michigan., Zondervan, 2000.
Grudem, Wayne A. SYSTEMATIC THEOLOGY, SECOND EDITION: An Introduction to Biblica Doctrine. S.L., Zondervan, 2020.
Hastings, Adrian. A World History of Christianity. 2000. Grand Rapids, Michigan., Eerdmans, 5 July 2000.
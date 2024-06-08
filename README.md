# Model-Fastapi

Building Python based API using FastAPI framework. Read documentation [here](https://fastapi.tiangolo.com/)

## Initial Requirements

Install dependency using `pip`

> _Run using python virtual environtment more recommended_

```python
pip install numpy tensorflow pandas fastapi uvicorn
```

## How to run ?

- Make sure you already installed all libraries needs
- Download model `H5` [here](https://drive.google.com/file/d/1AmsC4JdKICY33dZYf8JIg1y1vbIKDcnt/view?usp=sharing) *NOTED: this model is large in size*
- Rename your `H5` model similar with main.py -> `model.h5`
- Make sure your model in the same directory with your main.py
- run with this command in your terminal
  ```bash
  uvicorn main:app --reload
  ```

## Check API

Open interactive API documentation (provided by [Swagger UI](https://github.com/swagger-api/swagger-ui))

Go to link [http:127.0.0.1:8000/docs](http:127.0.0.1:8000/docs)

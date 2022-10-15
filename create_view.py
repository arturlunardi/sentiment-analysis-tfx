from google.cloud import bigquery


def _extract_json(column, feature_name):
  return "JSON_EXTRACT({}, '$.{}')".format(column, feature_name)


def _cast_to_numeric(field):
  return "CAST({} AS FLOAT64)".format(field)


def _add_alias(field, feature_name):
  return "{} AS {}".format(field, feature_name.replace('-prod', ''))


def create_view(GOOGLE_CLOUD_PROJECT, BQ_DATASET_NAME, BQ_TABLE_NAME, PIPELINE_NAME, VERSION_NAME):
    TITLE_FEATURES = ['title-prod']

    LABEL_KEY = 'label_key'
    SCORE_KEY = 'prediction_confidence'

    FEATURE_NAMES = TITLE_FEATURES

    view_name = "vw_"+BQ_TABLE_NAME+"_"+VERSION_NAME

    colum_names = FEATURE_NAMES

    json_features_extraction = []
    for feature_name in colum_names:
      field = _extract_json('instances', feature_name + '[0]')

      field = _add_alias(field, feature_name)
      json_features_extraction.append(field)

    json_features_extraction = ', \r\n    '.join(json_features_extraction)

    json_prediction_extraction = []

    for feature_name in [LABEL_KEY, SCORE_KEY]:
      field = _extract_json('predictions', feature_name)
      field = _cast_to_numeric(field)
      field = _add_alias(field, feature_name)
      json_prediction_extraction.append(field)

    json_prediction_extraction = ', \r\n    '.join(json_prediction_extraction)

    sql_script = '''
    CREATE OR REPLACE VIEW @dataset_name.@view_name
    AS
    SELECT 
        model, 
        model_version, 
        time,
     ARRAY(
          SELECT AS STRUCT
          @json_features_extraction, 
          @json_prediction_extraction
          FROM 
          UNNEST(JSON_EXTRACT_ARRAY(raw_prediction, "$.predictions")
          ) predictions WITH OFFSET AS f1 
          JOIN
          UNNEST(JSON_EXTRACT_ARRAY(raw_data, "$.instances")) instances WITH OFFSET AS f2
          ON f1=f2
      ) as request
    FROM 
    @project.@dataset_name.@table_name
    WHERE 
    model = "@model_name" AND
    model_version = "@version"
    '''

    sql_script = sql_script.replace("@project", GOOGLE_CLOUD_PROJECT)
    sql_script = sql_script.replace("@dataset_name", BQ_DATASET_NAME)
    sql_script = sql_script.replace("@table_name", BQ_TABLE_NAME)
    sql_script = sql_script.replace("@view_name", view_name)
    sql_script = sql_script.replace("@model_name", PIPELINE_NAME.replace('-', '_'))
    sql_script = sql_script.replace("@version", VERSION_NAME)
    sql_script = sql_script.replace("@json_features_extraction", json_features_extraction)
    sql_script = sql_script.replace("@json_prediction_extraction", json_prediction_extraction)

    client = bigquery.Client(GOOGLE_CLOUD_PROJECT)
    client.query(query = sql_script)
    print("View was created or replaced.")
    
    return sql_script
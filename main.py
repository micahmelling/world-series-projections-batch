import joblib
import yagmail
import random
import numpy as np
import pandas as pd
import ast

from datetime import datetime

from ds_helpers import aws
from modeling.train import assemble_modeling_data


def main():
    today = datetime.today().date().strftime('%Y-%m-%d')
    years = list(np.arange(1905, 2021))
    random_year = random.choice(years)
    pipeline = joblib.load('modeling/xgboost_20210528203348828936/models/model.pkl')
    teams_data = assemble_modeling_data()
    year_data = teams_data.loc[teams_data['team_yearID'] == random_year]
    predictions_df = pd.concat(
        [
            pd.DataFrame(pipeline.predict_proba(year_data), columns=['_', 'predicted_probability']),
            year_data[['teamIDwinner', 'team_teamID']].reset_index(drop=True)
        ],
        axis=1)
    predictions_df.drop('_', 1, inplace=True)
    predictions_df.rename(columns={'team_teamID': 'team', 'teamIDwinner': 'winner'}, inplace=True)
    predictions_df['winner'] = np.where(predictions_df['winner'] == 0, 'no', 'yes')
    predictions_df.to_csv(f'world_series_predictions_for_{random_year}.csv', index=False)

    email_creds_dict = aws.get_secrets_manager_secret('yagmail-credentials')
    username = email_creds_dict.get('username')
    password = email_creds_dict.get('password')
    email_recipients_dict = aws.get_secrets_manager_secret('ws-email-recipients')
    recipients = email_recipients_dict.get('recipients')
    recipients = ast.literal_eval(recipients)
    yag = yagmail.SMTP(username, password)
    yag.send(
        to=recipients,
        subject=f'World Series Predictions for {random_year} Produced on {today}',
        contents='Please see attachment for predictions.',
        attachments=[f'world_series_predictions_for_{random_year}.csv']
    )


if __name__ == "__main__":
    main()

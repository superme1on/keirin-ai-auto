# keirin-ai-auto

GitHub Actionsで自動起動する競輪予想AIスターターです。

## できること

- 毎朝、自動で予想CSVを作成
- 週1回、自動でモデル再学習
- データがない場合はサンプルデータで動作確認
- outputs/latest_predictions.csv に予想結果を保存
- outputs/index.html に簡易表示を保存

## 本物データを使う場合

GitHub Repository Settings > Secrets and variables > Actions で以下を設定してください。

- `KEIRIN_HISTORY_CSV_URL`
- `KEIRIN_TODAY_CSV_URL`

CSV形式は、最低限以下の列を想定しています。

### history.csv

```csv
race_id,date,venue,race_no,player_id,car_no,age,score,win_rate,place2_rate,place3_rate,back_count,style,recent_avg_finish,days_since_last_race,venue_win_rate,odds_win,finish_pos
```

### today_entries.csv

```csv
race_id,date,venue,race_no,player_id,car_no,age,score,win_rate,place2_rate,place3_rate,back_count,style,recent_avg_finish,days_since_last_race,venue_win_rate,odds_win
```

## 手動実行

```bash
pip install -r requirements.txt
python src/make_sample_data.py --if-missing
python src/train.py
python src/predict.py
```

## 注意

これは予想補助ツールです。的中や利益を保証するものではありません。

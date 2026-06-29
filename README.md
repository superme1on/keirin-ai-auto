# keirin-ai-auto

GitHub Actionsで自動起動する競輪予想AIスターターです。

## できること

- 毎朝、自動で予想CSVを作成
- 毎朝、最新の過去データでモデル再学習
- 週1回、定期的にモデル再学習
- データがない場合はサンプルデータで動作確認
- outputs/latest_predictions.csv に予想結果を保存
- outputs/latest_backtest.csv に過去データでの損益検証を保存
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

賭け金は1点100円で計算します。変更したい場合は `BET_STAKE_YEN` を指定してください。

```bash
BET_STAKE_YEN=500 python src/train.py
BET_STAKE_YEN=500 python src/predict.py
```

## 出力

`outputs/latest_predictions.csv` には以下の損益列が入ります。

- `stake_yen`: 1点あたりの賭け金
- `win_return_yen`: 当たった場合の払戻金
- `win_profit_yen`: 当たった場合の利益
- `loss_amount_yen`: 外れた場合の損失
- `expected_profit_yen`: 確率とオッズから見た期待利益

`outputs/backtest_summary.csv` は過去データのテスト期間で、実際に当たり外れを判定した損益サマリーです。

## 注意

これは予想補助ツールです。的中や利益を保証するものではありません。

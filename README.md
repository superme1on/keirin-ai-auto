# keirin-ai-auto

GitHub Actionsで自動起動する競輪予想AIスターターです。

## できること

- 毎朝、自動で予想CSVを作成
- 毎晩、決済後に最新の過去データでモデル再学習
- 週1回、定期的にモデル再学習
- WINTICKETの出走表を自動取得
- WINTICKETの過去結果から学習用 `history.csv` を自動構築
- データが取れない場合はサンプルデータで動作確認
- outputs/latest_predictions.csv に予想結果を保存
- outputs/latest_bets.csv に実際に買う複数券種の候補と購入額を保存
- outputs/latest_bet_candidates.csv に検討した複数券種の候補を保存
- outputs/japanese_report.md に日本語の購入理由と損益結果を保存
- outputs/growth_log.csv に学習精度と損益の成長記録を追記
- outputs/trial_report.md に直近31日の実購入ベース損益を保存
- outputs/latest_backtest.csv に過去データでの損益検証を保存
- outputs/profit_backtest_report.md に過去実オッズ込みの損益バックテストを保存
- outputs/multi_bet_backtest_report.md に券種別の損益バックテストを保存
- outputs/index.html に簡易表示を保存

## 本物データを使う場合

GitHub Repository Settings > Secrets and variables > Actions で以下を設定してください。

- `KEIRIN_HISTORY_CSV_URL`
- `KEIRIN_TODAY_CSV_URL`

`KEIRIN_TODAY_CSV_URL` が空の場合は、WINTICKETの出走表ページから今日の出走表を自動取得します。自動取得を止めたい場合は Actions の環境変数で `AUTO_FETCH_TODAY_ENTRIES=0` を指定してください。

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
python src/ingest.py
python src/build_history.py --months-back 24 --max-races 10000 --workers 8 --sleep-sec 0 --progress-every 250
python src/train.py
python src/profit_backtest.py --min-train-dates 30 --min-prob 0.10 --min-expected-profit 1200 --max-odds 200 --max-bets-per-race 2 --max-bets-per-day 40 --base-stake 100 --max-stake 500
python src/multi_bet_backtest.py --min-train-dates 30 --retrain-every-days 7 --top-k 5 --min-prob 0.02 --min-expected-profit 100 --max-odds 300 --max-bets-per-race-type 2 --max-bets-per-day-type 40 --base-stake 100 --max-stake 500
python src/fetch_today_entries.py
python src/predict.py
```

買い目は、券種別バックテストで絞った条件をデフォルトにしています。

- `BET_BASE_STAKE_YEN`: 最低購入額。標準は `100`
- `BET_MAX_STAKE_YEN`: 期待値が高いときの購入上限。標準は `500`

```bash
BET_BASE_STAKE_YEN=100 BET_MAX_STAKE_YEN=500 python src/predict.py
```

## 出力

`outputs/latest_predictions.csv` には以下の損益列が入ります。

- `stake_yen`: 1点あたりの賭け金
- `win_return_yen`: 当たった場合の払戻金
- `win_profit_yen`: 当たった場合の利益
- `loss_amount_yen`: 外れた場合の損失
- `expected_profit_yen`: 確率とオッズから見た期待利益

`outputs/backtest_summary.csv` は過去データのテスト期間で、実際に当たり外れを判定した損益サマリーです。

`outputs/latest_bets.csv` には、AIが実際に買う候補だけを出します。WINTICKETからオッズが取れた場合は、以下も入ります。

- `bet_type`: 券種
- `bet_label`: 日本語の券種名
- `buy`: 買い目
- `odds_used`: 判定に使ったオッズ
- `return_if_hit_yen`: 当たった場合の払戻金
- `expected_profit_yen`: AI確率とオッズから見た期待利益

`outputs/latest_bet_candidates.csv` は、AIが検討した全3連単候補です。

`outputs/purchase_plan.csv` は、`expected_profit_yen > 0` の買い目だけを抜き出した購入予定です。

`outputs/settled_bets.csv` と `outputs/settlement_summary.csv` は、レース結果取得後の損益集計です。

`outputs/japanese_report.md` は、購入条件、損益、主な買い目、選んだ理由を日本語でまとめたレポートです。

`outputs/growth_log.csv` は、学習時のAUC、本命的中率、結果精算後の損益を追記していく成長記録です。

`outputs/trial_report.md` は、直近31日の決済ログを実購入ベースで合算した1か月トライアル用レポートです。

`outputs/shadow_report.md` は、購入停止中に記録した影予想を公式払戻で精算した日別・券種別の成長記録です。`outputs/shadow_daily_summary.csv` に日別損益、`outputs/shadow_settled_bets.csv` に全買い目の勝敗を追記します。

`outputs/walk_forward_summary.csv` は、結果を見ずに日付順で予想し、結果が出たら次の日の学習に追加する実戦形式の検証です。

`outputs/profit_backtest_report.md` は、過去の実オッズを使った損益バックテストです。

`outputs/multi_bet_backtest_report.md` は、2車単・2車複・ワイド・3連複・3連単などを横並びで比較した損益バックテストです。

## 注意

これは予想補助ツールです。的中や利益を保証するものではありません。

## 厳密検証と購入ゲート

履歴作成では、落車・失格・途中棄権を含む全出走者を保持します。学習前に次の監査を実行し、出走者が1人でも欠けたレースがあれば失敗します。

```bash
python src/audit_history.py --fail-on-leakage
```

時間順の開発・校正・ホールドアウト検証は次で実行します。

```bash
python src/honest_backtest.py --rebuild-entry
```

`src/external_holdout.py` は、開発履歴とURLが重複しないレースを評価し、使用モデルのSHA-256を記録します。`outputs/external_holdout_overall.json` の `target_passed` が `true` になるまで、`outputs/latest_bets.csv` は0点です。未承認候補は `outputs/latest_shadow_bets.csv` にだけ保存されます。

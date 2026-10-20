# 初期設計とパラメータ集中探索

v005は、初期実験計画とパラメータ集中探索を `trial/focus_settings.json` で個別に有効化できます。設定ファイルがない場合、または3つの有効化フラグを省略した場合は、すべて有効です。明示されたfalseはそのまま適用されます。既存trialの実験CSV、過去のステップ、過去の予測結果は変更されません。設定が反映されるのは、その設定を使って新しく実行するtrialです。

## 設定例

```json
{
  "initial_design_enabled": true,
  "parameter_focus_enabled": true,
  "temporary_fix_enabled": true,
  "initial_min_conditions": 12,
  "initial_max_conditions": 48,
  "max_total_experiments": null,
  "confirmation_slots": 1,
  "stability_updates": 3,
  "effect_tolerances": {},
  "never_fix": [],
  "resamples": 12,
  "max_screening_samples": 256,
  "min_background_checks": 2,
  "global_check_interval": 3
}
```

全項目を省略したときの既定値は次のとおりです。上の設定例では3機能を有効にしています。`max_total_experiments`は3フラグがOFFでも指定した場合に適用されます。

| 項目 | 既定値 | 説明 |
|---|---:|---|
| `initial_design_enabled` | `true` | 初期実験を全入力の範囲を代表する計画にする |
| `parameter_focus_enabled` | `true` | 推定効果に応じて候補の優先度を調整し、探索を有力な入力範囲へ集中する |
| `temporary_fix_enabled` | `true` | 十分な確認を経た入力を一時固定し、定期的な再確認で必要なら探索へ戻す。`parameter_focus_enabled` も必要 |
| `initial_min_conditions` | `12` | 初期設計の目標件数の下限 |
| `initial_max_conditions` | `48` | 初期設計の目標件数の上限 |
| `max_total_experiments` | `null` | trial全体の実験数上限。`null` は追加上限なし |
| `confirmation_slots` | `1` | 次の提案枠のうち、再確認や固定候補の確認に割り当てる枠数 |
| `stability_updates` | `3` | 効果判定が安定したとみなす更新回数 |
| `effect_tolerances` | `{}` | 目的・制約の出力列名から、意味のある効果とみなす許容幅への対応。列の実単位で指定 |
| `never_fix` | `[]` | 一時固定の対象から除外する入力列名 |
| `resamples` | `12` | 効果の再標本化評価の回数 |
| `max_screening_samples` | `256` | 効果スクリーニングに使う最大サンプル数 |
| `min_background_checks` | `2` | 他の入力効果を確認するために保つ背景確認数 |
| `global_check_interval` | `3` | 全体探索を再確認する間隔（更新回数） |

`initial_min_conditions` は `initial_max_conditions` 以下にします。`max_total_experiments` は正の整数または `null` です。許容幅は有限の正数で、目的列または制約列の名前と、その出力の実単位で指定します。たとえば温度なら温度単位、寸法なら寸法単位です。許容幅が設定されていない出力については、効果の大きさだけを根拠に自動固定しません。制約を無視してよいという設定にはなりません。

## 有効化の組み合わせ

- 初期設計だけを使う場合は `initial_design_enabled: true`、集中探索2項目は `false` にします。
- 集中探索だけを使う場合は `initial_design_enabled: false`、`parameter_focus_enabled: true` にします。`temporary_fix_enabled: false` なら入力を固定せず、候補の重み付けによって探索を集中させます。
- 一時固定と再確認を使う場合は、`parameter_focus_enabled: true` と `temporary_fix_enabled: true` の両方を指定します。
- 3項目をすべて有効にすると、初期設計の後に集中探索を行います。

全機能を無効にする場合は3フラグをすべてfalseにします。集中探索を無効にする場合は、一時固定もfalseにします。初期設計と集中探索は独立して切り替えられます。設定を変えても、既に作成済みのtrialの履歴やデータを書き換えません。

## 判断の読み方と限界

効果推定は、保存された入力・出力データに対するNumPyの回帰ベースのヒューリスティックです。信頼区間や統計的な保証を与えるものではありません。入力の効果を単純に比較しにくい条件では、状態が `UNDECIDED` となり、判定を保留することがあります。しきい値が未設定の出力については、`FIX_CANDIDATE` から自動的な固定判断に進めません。一時固定が有効でも、再確認のための背景点と全体確認を設け、効果が変わった入力は再び探索対象に戻せる設計です。

GUIは各予測runに対応する `output/parameter_focus/run_XXXX.json` があれば、候補範囲と各入力の状態を表示します。対応はファイル名と応答CSVの `run_id` が完全一致するものを使います。候補範囲が `focused_with_checks` の予測面は、集中探索と確認点を含む範囲内の断面です。全候補域での最適性を示すものではありません。GUIはこのJSONと保存済みCSVを読むだけで、学習や探索、trialの変更は行いません。

NPZ入力データは元の入力を保持します。探索CLIとシミュレータは別プロセスで実行し、推薦された入力条件と実測結果をファイルで接続します。初期設計を含め、v005からシミュレータを起動しません。

## 初期実験と通常の操作

`--new`で3フラグがONの`focus_settings.json`も作成します。`--prepare`でCSVヘッダーを用意してから`--run --n 5`などを実行します。実験CSVが空でも、最大5件の入力条件を推薦します。予測欄は空欄で、結果は生成しません。実測を追記して再実行してください。

初期件数の目安は操作変数数dに対して`max(initial_min_conditions, 2*d+4)`を4件単位へそろえた値です。有限格子の件数と`initial_max_conditions`で上限を設けます。水準の被覆・範囲の広さ・設計行列のrank・補助回帰の未使用点誤差が不足する場合には延長します。上限でも不明な作用は保留して探索へ進めます。初期段階が終了した同じ問題・流入状態では、確認実験は探索側が担当します。

初期条件、改善条件、確認条件は、すべて`--n`の最大件数と実験予算に含まれます。総予算は実験CSVの実測行数で数えます。既存の`max_additional_experiments`も継続して適用し、途中で初期件数や固定状態が変わっても起点を再設定しません。初回runにデータがなければ追加予算の起点は0です。既存データから開始する場合は従来どおり開始時の行数を起点にします。

## 一時固定の条件と保存内容

目的値とすべての制約に許容作用量を指定し、補助回帰の作用・誤差・ノイズ目安が許容幅未満であること、制約余裕、異なるデータ更新での安定、複数背景での離散水準比較、現在の良い条件の背景での比較を必要とします。同一データの再実行は安定回数を増やしません。データ訂正・削除・反復平均の変化は、独立条件の追加と区別します。

一時固定の数値入力は最大16水準に限定します。カテゴリ入力が変動する場合、現版の補助回帰では自動固定を保留します。固定後の作用増大、制約余裕不足、設定・流入状態の変更で再評価します。確認点のHybrid予測が補助回帰と食い違う場合も、次runの固定を保留します。効果の小ささは確認した範囲での診断であり、全域の無効性の証明ではありません。

最新状態は`output/parameter_focus.json`、時点別状態は`output/parameter_focus/run_XXXX.json`、初期段階は`output/initial_design_status.json`に保存します。推薦CSVの`selection_role`に`INITIAL_DESIGN`、`SCREENING`、`FROZEN_CHECK`、`GLOBAL_CHECK`を記録します。後3種は改善EIとは別基準で選び、禁止領域と予測合格確率の下限は守ります。

## 停止と計算負荷

既存の停止条件を緩和しません。縮小した候補断面を全列挙しても、元の全候補域を調べたとは記録しません。確認・改善候補がなく、全域収束の根拠が不足する場合は`REVIEW_REQUIRED`です。現版には、限定断面の収束を全域収束に置き換える経路はありません。

補助回帰はNumPyの正則化最小二乗と少数のデータ抜き取りを使い、入力ごとのNN再学習は行いません。既定は256分析条件・12回の抜き取りで、設定上限は512条件・64回です。改善と確認を合成した候補総数に既存の候補予算を適用し、学習・スレッド・距離計算の上限を維持します。推薦、停止、保存用に同じ予測を共有します。

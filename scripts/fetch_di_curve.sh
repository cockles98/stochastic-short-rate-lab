DATA_URL='https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados?formato=json'
OUT_FILE='data/raw_di_curve.json'

declare -A HEADERS=( ["Accept"]="application/json" )

echo "Baixando curva DI do Bacen..."
curl -s -H "Accept: application/json" "$DATA_URL" -o "$OUT_FILE"

if [ ! -s "$OUT_FILE" ]; then
  echo "Download falhou ou arquivo vazio" >&2
  exit 1
fi

echo "Salvo em $OUT_FILE"

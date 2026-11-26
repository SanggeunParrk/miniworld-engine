# usage: ab2.sh L "QS" "MACROS" ... pairs: each variant = QS(0/1) then macros ("-" none)
L=$1; shift
while [ $# -ge 2 ]; do
 q=$1; v=$2; [ "$v" = "-" ] && v=""; shift 2
 echo "QS=$q $v: $(env FOLDJET_ENGINE_TRIATTN_B200_QS=$q $W/b2511/dev.sh -x -t 600 n7-triattn python $W/b2511/work/n7-triattn/abp7.py "$v" $L '{"d_pair":128,"n_head":4,"starting":true}' 2>&1 | grep "^L[0-9]")"
done

from pathlib import Path
import re
p=Path(__file__).resolve().parent
for src in ['front_cluster_pipe','front_cluster512']:
 s=(p/(src+'.cu')).read_text();a=s.index('TMN_DEVI void consumer');b=s.index('TMN_DEVI void reduce_at',a);f=s[a:b]
 release='if(threadIdx.x==0){if(tile+CLUSTERS*4<p.tiles)mbar_arrive_expect_tx(&b->copy,131072);for(int group=0;group<4;++group)remote_arrive(b->free+slot,group);}'
 assert f.count(release)==1;f=f.replace(release,'')
 start=f.index('  if(threadIdx.x==0){mbar_arrive_expect_tx(bar,32768);');end=f.index('\n }\n p.partln') if '\n }\n p.partln' in f else f.index('\n }\n if(threadIdx.x<256)p.partln')
 f=f[:start]+'  '+release+'uint8_t* lnsm=sm+131072;\n'+re.sub(r'\bsm\b','lnsm',f[start:end])+f[end:];s=s[:a]+f+s[b:]
 name=src+'_early';(p/(name+'.cu')).write_text(s);(p/(name+'.launch.json')).write_text((p/(src+'.launch.json')).read_text())

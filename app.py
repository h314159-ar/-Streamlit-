# poetry_land — 诗词交互探索平台
import streamlit as st
import pandas as pd
import random as _random
from collections import defaultdict, Counter
import plotly.express as px
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

st.set_page_config(page_title="poetry_land · 诗词域", page_icon="🖋", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

  .stApp {
    background-color: #F5F0E8;
    font-family: "Noto Sans SC", -apple-system, sans-serif;
    color: #2C2C2C;
  }
  .main > div { padding: 1.5rem 2.5rem; }
  .block-container { max-width: 1200px; padding-top: 1.5rem; }

  h1, h2, h3 {
    font-family: "Noto Serif SC", Georgia, serif;
    color: #3A2A1A !important;
    font-weight: 700;
  }
  h1 {
    font-size: 2.4rem;
    border-bottom: 2px solid #C04040;
    padding-bottom: 0.6rem;
    margin-bottom: 1.2rem;
    letter-spacing: 0.05em;
  }
  h2 { font-size: 1.6rem; margin-top: 1.5rem; }
  h3 { font-size: 1.2rem; }
  .stMarkdown, .stText { color: #3A2A1A; }
  p { line-height: 1.8; }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EDE7DC 0%, #E8E0D2 100%);
    border-right: 1px solid #D4C5A9;
    padding: 1rem 0;
  }
  section[data-testid="stSidebar"] .stRadio label {
    font-family: "Noto Serif SC", serif;
    font-size: 1.05rem;
    padding: 0.6rem 1.2rem;
    border-radius: 6px;
    transition: all 0.2s;
  }
  section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(192, 64, 64, 0.08);
  }

  .stButton>button {
    background-color: #C04040 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px;
    padding: 0.35rem 1.5rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    transition: all 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  .stButton>button:hover {
    background-color: #A03030 !important;
    transform: translateY(-1px);
    box-shadow: 0 3px 8px rgba(192,64,64,0.2);
  }
  .stButton>button:active { transform: scale(0.97); }

  .stTextInput>div>div>input, .stTextArea>div>div>textarea {
    border-radius: 6px; border: 1px solid #D4C5A9;
    background: #FCFAF5; padding: 0.5rem 0.8rem;
    font-family: "Noto Sans SC", sans-serif;
  }
  .stTextInput>div>div>input:focus { border-color: #C04040; }
  .stSelectbox>div>div { border-radius: 6px; border: 1px solid #D4C5A9; background: #FCFAF5; }

  div[data-testid="stMetricValue"] {
    font-size: 2rem; color: #3A2A1A;
    font-family: "Noto Serif SC", serif;
  }
  div[data-testid="stMetricLabel"] {
    color: #8B7355; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.05em;
  }

  .poem-card {
    background: linear-gradient(135deg, #FCFAF5 0%, #FFFCF7 100%);
    border: 1px solid #D4C5A9; border-radius: 8px;
    padding: 1.5rem 2rem; margin: 0.8rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    position: relative;
  }
  .poem-card::before {
    content: "\\2726";
    position: absolute; top: 8px; right: 14px;
    color: #C04040; font-size: 0.8rem; opacity: 0.3;
  }

  .ai-reply {
    background: #FCFAF5;
    border-left: 4px solid #C04040;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    font-family: "Noto Serif SC", serif;
  }
  .ai-reply b { color: #8B4513; }

  .source-tag {
    display: inline-block; padding: 0.15rem 0.6rem;
    border-radius: 9999px; font-size: 0.75rem;
    letter-spacing: 0.05em; background: #FDEBEC; color: #9F2F2D;
  }

  .stDataFrame { border: 1px solid #D4C5A9; border-radius: 8px; overflow: hidden; }
  .stDataFrame thead th {
    background: #EDE7DC; color: #3A2A1A; font-weight: 600;
    padding: 0.6rem 0.8rem;
  }
  .stDataFrame tbody td { padding: 0.5rem 0.8rem; border-bottom: 1px solid #EDE7DC; }

  .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #D4C5A9; }
  .stTabs [data-baseweb="tab"] {
    font-family: "Noto Serif SC", serif;
    padding: 0.6rem 1.4rem;
    border-radius: 6px 6px 0 0;
  }
  .stTabs [aria-selected="true"] {
    background: #FCFAF5; border: 1px solid #D4C5A9;
    border-bottom: 2px solid #C04040; color: #C04040;
  }

  .stExpander { border: 1px solid #D4C5A9; border-radius: 8px; background: #FCFAF5; }

  .stProgress > div > div {
    background: linear-gradient(90deg, #6B8E23, #8DB600) !important;
  }

  hr { border: none; border-top: 1px solid #D4C5A9; margin: 1.5rem 0; }
  .stAlert { border-radius: 6px; border: none; border-left: 4px solid; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #F5F0E8; }
  ::-webkit-scrollbar-thumb { background: #D4C5A9; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #C04040; }
</style>
""", unsafe_allow_html=True)
POEMS = [
    {
        "title": "静夜思",
        "author": "李白",
        "dynasty": "唐",
        "content": "床前明月光 疑是地上霜 举头望明月 低头思故乡",
        "style": "豪放"
    },
    {
        "title": "望庐山瀑布",
        "author": "李白",
        "dynasty": "唐",
        "content": "日照香炉生紫烟 遥看瀑布挂前川 飞流直下三千尺 疑是银河落九天",
        "style": "豪放"
    },
    {
        "title": "早发白帝城",
        "author": "李白",
        "dynasty": "唐",
        "content": "朝辞白帝彩云间 千里江陵一日还 两岸猿声啼不住 轻舟已过万重山",
        "style": "豪放"
    },
    {
        "title": "将进酒",
        "author": "李白",
        "dynasty": "唐",
        "content": "君不见黄河之水天上来 奔流到海不复回 君不见高堂明镜悲白发 朝如青丝暮成雪 人生得意须尽欢 莫使金樽空对月 天生我材必有用 千金散尽还复来",
        "style": "豪放"
    },
    {
        "title": "行路难",
        "author": "李白",
        "dynasty": "唐",
        "content": "金樽清酒斗十千 玉盘珍羞直万钱 停杯投箸不能食 拔剑四顾心茫然 欲渡黄河冰塞川 将登太行雪满山 长风破浪会有时 直挂云帆济沧海",
        "style": "豪放"
    },
    {
        "title": "登金陵凤凰台",
        "author": "李白",
        "dynasty": "唐",
        "content": "凤凰台上凤凰游 凤去台空江自流 吴宫花草埋幽径 晋代衣冠成古丘 三山半落青天外 二水中分白鹭洲 总为浮云能蔽日 长安不见使人愁",
        "style": "豪放"
    },
    {
        "title": "送友人",
        "author": "李白",
        "dynasty": "唐",
        "content": "青山横北郭 白水绕东城 此地一为别 孤蓬万里征 浮云游子意 落日故人情 挥手自兹去 萧萧班马鸣",
        "style": "豪放"
    },
    {
        "title": "月下独酌",
        "author": "李白",
        "dynasty": "唐",
        "content": "花间一壶酒 独酌无相亲 举杯邀明月 对影成三人 月既不解饮 影徒随我身 暂伴月将影 行乐须及春",
        "style": "豪放"
    },
    {
        "title": "望天门山",
        "author": "李白",
        "dynasty": "唐",
        "content": "天门中断楚江开 碧水东流至此回 两岸青山相对出 孤帆一片日边来",
        "style": "豪放"
    },
    {
        "title": "赠汪伦",
        "author": "李白",
        "dynasty": "唐",
        "content": "李白乘舟将欲行 忽闻岸上踏歌声 桃花潭水深千尺 不及汪伦送我情",
        "style": "豪放"
    },
    {
        "title": "念奴娇·赤壁怀古",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "大江东去浪淘尽千古风流人物 故垒西边人道是三国周郎赤壁 乱石穿空惊涛拍岸卷起千堆雪 江山如画一时多少豪杰",
        "style": "豪放"
    },
    {
        "title": "水调歌头·明月几时有",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "明月几时有把酒问青天 不知天上宫阙今夕是何年 我欲乘风归去又恐琼楼玉宇高处不胜寒 起舞弄清影何似在人间",
        "style": "豪放"
    },
    {
        "title": "定风波",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "莫听穿林打叶声何妨吟啸且徐行 竹杖芒鞋轻胜马谁怕一蓑烟雨任平生 料峭春风吹酒醒微冷山头斜照却相迎 回首向来萧瑟处归去也无风雨也无晴",
        "style": "豪放"
    },
    {
        "title": "江城子·密州出猎",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "老夫聊发少年狂左牵黄右擎苍 锦帽貂裘千骑卷平冈 为报倾城随太守亲射虎看孙郎 酒酣胸胆尚开张鬓微霜又何妨 持节云中何日遣冯唐 会挽雕弓如满月西北望射天狼",
        "style": "豪放"
    },
    {
        "title": "临江仙",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "夜饮东坡醒复醉归来仿佛三更 家童鼻息已雷鸣敲门都不应倚杖听江声 长恨此身非我有何时忘却营营 夜阑风静縠纹平小舟从此逝江海寄余生",
        "style": "豪放"
    },
    {
        "title": "题西林壁",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "横看成岭侧成峰 远近高低各不同 不识庐山真面目 只缘身在此山中",
        "style": "豪放"
    },
    {
        "title": "饮湖上初晴后雨",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "水光潋滟晴方好 山色空蒙雨亦奇 欲把西湖比西子 淡妆浓抹总相宜",
        "style": "豪放"
    },
    {
        "title": "破阵子·为陈同甫赋壮词以寄之",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "醉里挑灯看剑梦回吹角连营 八百里分麾下炙五十弦翻塞外声沙场秋点兵 马作的卢飞快弓如霹雳弦惊 了却君王天下事赢得生前身后名可怜白发生",
        "style": "豪放"
    },
    {
        "title": "永遇乐·京口北固亭怀古",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "千古江山英雄无觅孙仲谋处 舞榭歌台风流总被雨打风吹去 斜阳草树寻常巷陌人道寄奴曾住 想当年金戈铁马气吞万里如虎",
        "style": "豪放"
    },
    {
        "title": "青玉案·元夕",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "东风夜放花千树更吹落星如雨 宝马雕车香满路凤箫声动玉壶光转一夜鱼龙舞 蛾儿雪柳黄金缕笑语盈盈暗香去 众里寻他千百度蓦然回首那人却在灯火阑珊处",
        "style": "豪放"
    },
    {
        "title": "南乡子·登京口北固亭有怀",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "何处望神州满眼风光北固楼 千古兴亡多少事悠悠不尽长江滚滚流 年少万兜鍪坐断东南战未休 天下英雄谁敌手曹刘生子当如孙仲谋",
        "style": "豪放"
    },
    {
        "title": "丑奴儿·书博山道中壁",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "少年不识愁滋味爱上层楼 爱上层楼为赋新词强说愁 而今识尽愁滋味欲说还休 欲说还休却道天凉好个秋",
        "style": "豪放"
    },
    {
        "title": "秋词",
        "author": "刘禹锡",
        "dynasty": "唐",
        "content": "自古逢秋悲寂寥 我言秋日胜春朝 晴空一鹤排云上 便引诗情到碧霄",
        "style": "豪放"
    },
    {
        "title": "浪淘沙",
        "author": "刘禹锡",
        "dynasty": "唐",
        "content": "九曲黄河万里沙 浪淘风簸自天涯 如今直上银河去 同到牵牛织女家",
        "style": "豪放"
    },
    {
        "title": "出塞",
        "author": "王昌龄",
        "dynasty": "唐",
        "content": "秦时明月汉时关 万里长征人未还 但使龙城飞将在 不教胡马度阴山",
        "style": "豪放"
    },
    {
        "title": "从军行",
        "author": "王昌龄",
        "dynasty": "唐",
        "content": "青海长云暗雪山 孤城遥望玉门关 黄沙百战穿金甲 不破楼兰终不还",
        "style": "豪放"
    },
    {
        "title": "凉州词",
        "author": "王翰",
        "dynasty": "唐",
        "content": "葡萄美酒夜光杯 欲饮琵琶马上催 醉卧沙场君莫笑 古来征战几人回",
        "style": "豪放"
    },
    {
        "title": "凉州词",
        "author": "王之涣",
        "dynasty": "唐",
        "content": "黄河远上白云间 一片孤城万仞山 羌笛何须怨杨柳 春风不度玉门关",
        "style": "豪放"
    },
    {
        "title": "登鹳雀楼",
        "author": "王之涣",
        "dynasty": "唐",
        "content": "白日依山尽 黄河入海流 欲穷千里目 更上一层楼",
        "style": "豪放"
    },
    {
        "title": "望岳",
        "author": "杜甫",
        "dynasty": "唐",
        "content": "岱宗夫如何 齐鲁青未了 造化钟神秀 阴阳割昏晓 荡胸生曾云 决眦入归鸟 会当凌绝顶 一览众山小",
        "style": "豪放"
    },
    {
        "title": "春望",
        "author": "杜甫",
        "dynasty": "唐",
        "content": "国破山河在 城春草木深 感时花溅泪 恨别鸟惊心 烽火连三月 家书抵万金 白头搔更短 浑欲不胜簪",
        "style": "豪放"
    },
    {
        "title": "满江红",
        "author": "岳飞",
        "dynasty": "宋",
        "content": "怒发冲冠凭栏处潇潇雨歇 抬望眼仰天长啸壮怀激烈 三十功名尘与土八千里路云和月 莫等闲白了少年头空悲切",
        "style": "豪放"
    },
    {
        "title": "观沧海",
        "author": "曹操",
        "dynasty": "汉",
        "content": "东临碣石以观沧海 水何澹澹山岛竦峙 树木丛生百草丰茂 秋风萧瑟洪波涌起 日月之行若出其中 星汉灿烂若出其里",
        "style": "豪放"
    },
    {
        "title": "短歌行",
        "author": "曹操",
        "dynasty": "汉",
        "content": "对酒当歌人生几何 譬如朝露去日苦多 慨当以慷忧思难忘 何以解忧唯有杜康 青青子衿悠悠我心 但为君故沉吟至今",
        "style": "豪放"
    },
    {
        "title": "龟虽寿",
        "author": "曹操",
        "dynasty": "汉",
        "content": "神龟虽寿犹有竟时 腾蛇乘雾终为土灰 老骥伏枥志在千里 烈士暮年壮心不已 盈缩之期不但在天 养怡之福可得永年",
        "style": "豪放"
    },
    {
        "title": "别董大",
        "author": "高适",
        "dynasty": "唐",
        "content": "千里黄云白日曛 北风吹雁雪纷纷 莫愁前路无知己 天下谁人不识君",
        "style": "豪放"
    },
    {
        "title": "燕歌行",
        "author": "高适",
        "dynasty": "唐",
        "content": "汉家烟尘在东北汉将辞家破残贼 男儿本自重横行天子非常赐颜色 从金伐鼓下榆关旌旆逶迤碣石间 校尉羽书飞瀚海单于猎火照狼山",
        "style": "豪放"
    },
    {
        "title": "白雪歌送武判官归京",
        "author": "岑参",
        "dynasty": "唐",
        "content": "北风卷地白草折胡天八月即飞雪 忽如一夜春风来千树万树梨花开 散入珠帘湿罗幕狐裘不暖锦衾薄 将军角弓不得控都护铁衣冷难着",
        "style": "豪放"
    },
    {
        "title": "忆江南",
        "author": "白居易",
        "dynasty": "唐",
        "content": "江南好风景旧曾谙 日出江花红胜火春来江水绿如蓝 能不忆江南",
        "style": "豪放"
    },
    {
        "title": "塞下曲",
        "author": "卢纶",
        "dynasty": "唐",
        "content": "月黑雁飞高 单于夜遁逃 欲将轻骑逐 大雪满弓刀",
        "style": "豪放"
    },
    {
        "title": "使至塞上",
        "author": "王维",
        "dynasty": "唐",
        "content": "单车欲问边 属国过居延 征蓬出汉塞 归雁入胡天 大漠孤烟直 长河落日圆 萧关逢候骑 都护在燕然",
        "style": "豪放"
    },
    {
        "title": "渡荆门送别",
        "author": "李白",
        "dynasty": "唐",
        "content": "渡远荆门外 来从楚国游 山随平野尽 江入大荒流 月下飞天镜 云生结海楼 仍怜故乡水 万里送行舟",
        "style": "豪放"
    },
    {
        "title": "黄鹤楼送孟浩然之广陵",
        "author": "李白",
        "dynasty": "唐",
        "content": "故人西辞黄鹤楼 烟花三月下扬州 孤帆远影碧空尽 唯见长江天际流",
        "style": "豪放"
    },
    {
        "title": "闻王昌龄左迁龙标遥有此寄",
        "author": "李白",
        "dynasty": "唐",
        "content": "杨花落尽子规啼 闻道龙标过五溪 我寄愁心与明月 随君直到夜郎西",
        "style": "豪放"
    },
    {
        "title": "登高",
        "author": "杜甫",
        "dynasty": "唐",
        "content": "风急天高猿啸哀 渚清沙白鸟飞回 无边落木萧萧下 不尽长江滚滚来 万里悲秋常作客 百年多病独登台 艰难苦恨繁霜鬓 潦倒新停浊酒杯",
        "style": "豪放"
    },
    {
        "title": "蜀道难",
        "author": "李白",
        "dynasty": "唐",
        "content": "噫吁嚱危乎高哉蜀道之难难于上青天 蚕丛及鱼凫开国何茫然 尔来四万八千岁不与秦塞通人烟 西当太白有鸟道可以横绝峨眉巅",
        "style": "豪放"
    },
    {
        "title": "梦游天姥吟留别",
        "author": "李白",
        "dynasty": "唐",
        "content": "海客谈瀛洲烟涛微茫信难求 越人语天姥云霞明灭或可睹 天姥连天向天横势拔五岳掩赤城 天台四万八千丈对此欲倒东南倾",
        "style": "豪放"
    },
    {
        "title": "宣州谢朓楼饯别校书叔云",
        "author": "李白",
        "dynasty": "唐",
        "content": "弃我去者昨日之日不可留 乱我心者今日之日多烦忧 长风万里送秋色对此可以酣高楼 蓬莱文章建安骨中间小谢又清发",
        "style": "豪放"
    },
    {
        "title": "侠客行",
        "author": "李白",
        "dynasty": "唐",
        "content": "赵客缦胡缨吴钩霜雪明 银鞍照白马飒沓如流星 十步杀一人千里不留行 事了拂衣去深藏身与名",
        "style": "豪放"
    },
    {
        "title": "登幽州台歌",
        "author": "陈子昂",
        "dynasty": "唐",
        "content": "前不见古人 后不见来者 念天地之悠悠 独怆然而涕下",
        "style": "豪放"
    },
    {
        "title": "南园十三首",
        "author": "李贺",
        "dynasty": "唐",
        "content": "男儿何不带吴钩 收取关山五十州 请君暂上凌烟阁 若个书生万户侯",
        "style": "豪放"
    },
    {
        "title": "雁门太守行",
        "author": "李贺",
        "dynasty": "唐",
        "content": "黑云压城城欲摧甲光向日金鳞开 角声满天秋色里塞上燕脂凝夜紫 半卷红旗临易水霜重鼓寒声不起 报君黄金台上意提携玉龙为君死",
        "style": "豪放"
    },
    {
        "title": "己亥杂诗",
        "author": "龚自珍",
        "dynasty": "清",
        "content": "浩荡离愁白日斜 吟鞭东指即天涯 落红不是无情物 化作春泥更护花",
        "style": "豪放"
    },
    {
        "title": "竹石",
        "author": "郑燮",
        "dynasty": "清",
        "content": "咬定青山不放松 立根原在破岩中 千磨万击还坚劲 任尔东西南北风",
        "style": "豪放"
    },
    {
        "title": "石灰吟",
        "author": "于谦",
        "dynasty": "明",
        "content": "千锤万凿出深山 烈火焚烧若等闲 粉骨碎身浑不怕 要留清白在人间",
        "style": "豪放"
    },
    {
        "title": "墨梅",
        "author": "王冕",
        "dynasty": "元",
        "content": "吾家洗砚池头树 朵朵花开淡墨痕 不要人夸好颜色 只留清气满乾坤",
        "style": "豪放"
    },
    {
        "title": "狱中题壁",
        "author": "谭嗣同",
        "dynasty": "清",
        "content": "望门投止思张俭 忍死须臾待杜根 我自横刀向天笑 去留肝胆两昆仑",
        "style": "豪放"
    },
    {
        "title": "对酒",
        "author": "秋瑾",
        "dynasty": "清",
        "content": "不惜千金买宝刀 貂裘换酒也堪豪 一腔热血勤珍重 洒去犹能化碧涛",
        "style": "豪放"
    },
    {
        "title": "泊秦淮",
        "author": "杜牧",
        "dynasty": "唐",
        "content": "烟笼寒水月笼沙 夜泊秦淮近酒家 商女不知亡国恨 隔江犹唱后庭花",
        "style": "豪放"
    },
    {
        "title": "赤壁",
        "author": "杜牧",
        "dynasty": "唐",
        "content": "折戟沉沙铁未销 自将磨洗认前朝 东风不与周郎便 铜雀春深锁二乔",
        "style": "豪放"
    },
    {
        "title": "过零丁洋",
        "author": "文天祥",
        "dynasty": "宋",
        "content": "辛苦遭逢起一经 干戈寥落四周星 山河破碎风飘絮 身世浮沉雨打萍 惶恐滩头说惶恐 零丁洋里叹零丁 人生自古谁无死 留取丹心照汗青",
        "style": "豪放"
    },
    {
        "title": "夏日绝句",
        "author": "李清照",
        "dynasty": "宋",
        "content": "生当作人杰 死亦为鬼雄 至今思项羽 不肯过江东",
        "style": "豪放"
    },
    {
        "title": "十一月四日风雨大作",
        "author": "陆游",
        "dynasty": "宋",
        "content": "僵卧孤村不自哀 尚思为国戍轮台 夜阑卧听风吹雨 铁马冰河入梦来",
        "style": "豪放"
    },
    {
        "title": "示儿",
        "author": "陆游",
        "dynasty": "宋",
        "content": "死去元知万事空 但悲不见九州同 王师北定中原日 家祭无忘告乃翁",
        "style": "豪放"
    },
    {
        "title": "书愤",
        "author": "陆游",
        "dynasty": "宋",
        "content": "早岁那知世事艰 中原北望气如山 楼船夜雪瓜洲渡 铁马秋风大散关 塞上长城空自许 镜中衰鬓已先斑 出师一表真名世 千载谁堪伯仲间",
        "style": "豪放"
    },
    {
        "title": "游山西村",
        "author": "陆游",
        "dynasty": "宋",
        "content": "莫笑农家腊酒浑 丰年留客足鸡豚 山重水复疑无路 柳暗花明又一村 箫鼓追随春社近 衣冠简朴古风存 从今若许闲乘月 拄杖无时夜叩门",
        "style": "豪放"
    },
    {
        "title": "渔家傲·秋思",
        "author": "范仲淹",
        "dynasty": "宋",
        "content": "塞下秋来风景异衡阳雁去无留意 四面边声连角起千嶂里长烟落日孤城闭 浊酒一杯家万里燕然未勒归无计 羌管悠悠霜满地人不寐将军白发征夫泪",
        "style": "豪放"
    },
    {
        "title": "江城子·乙卯正月二十日夜记梦",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "十年生死两茫茫不思量自难忘 千里孤坟无处话凄凉 纵使相逢应不识尘满面鬓如霜 夜来幽梦忽还乡小轩窗正梳妆 相顾无言惟有泪千行 料得年年肠断处明月夜短松冈",
        "style": "豪放"
    },
    {
        "title": "声声慢",
        "author": "李清照",
        "dynasty": "宋",
        "content": "寻寻觅觅冷冷清清凄凄惨惨戚戚 乍暖还寒时候最难将息 三杯两盏淡酒怎敌他晚来风急 雁过也正伤心却是旧时相识",
        "style": "婉约"
    },
    {
        "title": "一剪梅",
        "author": "李清照",
        "dynasty": "宋",
        "content": "红藕香残玉簟秋轻解罗裳独上兰舟 云中谁寄锦书来雁字回时月满西楼 花自飘零水自流一种相思两处闲愁 此情无计可消除才下眉头却上心头",
        "style": "婉约"
    },
    {
        "title": "醉花阴",
        "author": "李清照",
        "dynasty": "宋",
        "content": "薄雾浓云愁永昼瑞脑消金兽 佳节又重阳玉枕纱厨半夜凉初透 东篱把酒黄昏后有暗香盈袖 莫道不销魂帘卷西风人比黄花瘦",
        "style": "婉约"
    },
    {
        "title": "武陵春",
        "author": "李清照",
        "dynasty": "宋",
        "content": "风住尘香花已尽日晚倦梳头 物是人非事事休欲语泪先流 闻说双溪春尚好也拟泛轻舟 只恐双溪舴艋舟载不动许多愁",
        "style": "婉约"
    },
    {
        "title": "凤凰台上忆吹箫",
        "author": "李清照",
        "dynasty": "宋",
        "content": "香冷金猊被翻红浪起来慵自梳头 任宝奁尘满日上帘钩 生怕离怀别苦多少事欲说还休 新来瘦非干病酒不是悲秋",
        "style": "婉约"
    },
    {
        "title": "雨霖铃",
        "author": "柳永",
        "dynasty": "宋",
        "content": "寒蝉凄切对长亭晚骤雨初歇 都门帐饮无绪留恋处兰舟催发 执手相看泪眼竟无语凝噎 念去去千里烟波暮霭沉沉楚天阔",
        "style": "婉约"
    },
    {
        "title": "望海潮",
        "author": "柳永",
        "dynasty": "宋",
        "content": "东南形胜三吴都会钱塘自古繁华 烟柳画桥风帘翠幕参差十万人家 云树绕堤沙怒涛卷霜雪天堑无涯 市列珠玑户盈罗绮竞豪奢",
        "style": "婉约"
    },
    {
        "title": "蝶恋花",
        "author": "柳永",
        "dynasty": "宋",
        "content": "伫倚危楼风细细望极春愁黯黯生天际 草色烟光残照里无言谁会凭阑意 拟把疏狂图一醉对酒当歌强乐还无味 衣带渐宽终不悔为伊消得人憔悴",
        "style": "婉约"
    },
    {
        "title": "菩萨蛮",
        "author": "温庭筠",
        "dynasty": "唐",
        "content": "小山重叠金明灭鬓云欲度香腮雪 懒起画蛾眉弄妆梳洗迟 照花前后镜花面交相映 新帖绣罗襦双双金鹧鸪",
        "style": "婉约"
    },
    {
        "title": "更漏子",
        "author": "温庭筠",
        "dynasty": "唐",
        "content": "玉炉香红蜡泪偏照画堂秋思 眉翠薄鬓云残夜长衾枕寒 梧桐树三更雨不道离情正苦 一叶叶一声声空阶滴到明",
        "style": "婉约"
    },
    {
        "title": "浣溪沙",
        "author": "晏殊",
        "dynasty": "宋",
        "content": "一曲新词酒一杯去年天气旧亭台夕阳西下几时回 无可奈何花落去似曾相识燕归来小园香径独徘徊",
        "style": "婉约"
    },
    {
        "title": "蝶恋花",
        "author": "晏殊",
        "dynasty": "宋",
        "content": "槛菊愁烟兰泣露罗幕轻寒燕子双飞去 明月不谙离恨苦斜光到晓穿朱户 昨夜西风凋碧树独上高楼望尽天涯路 欲寄彩笺兼尺素山长水阔知何处",
        "style": "婉约"
    },
    {
        "title": "鹊桥仙",
        "author": "秦观",
        "dynasty": "宋",
        "content": "纤云弄巧飞星传恨银汉迢迢暗度 金风玉露一相逢便胜却人间无数 柔情似水佳期如梦忍顾鹊桥归路 两情若是久长时又岂在朝朝暮暮",
        "style": "婉约"
    },
    {
        "title": "踏莎行",
        "author": "秦观",
        "dynasty": "宋",
        "content": "雾失楼台月迷津渡桃源望断无寻处 可堪孤馆闭春寒杜鹃声里斜阳暮 驿寄梅花鱼传尺素砌成此恨无重数 郴江幸自绕郴山为谁流下潇湘去",
        "style": "婉约"
    },
    {
        "title": "虞美人",
        "author": "李煜",
        "dynasty": "五代",
        "content": "春花秋月何时了往事知多少 小楼昨夜又东风故国不堪回首月明中 雕栏玉砌应犹在只是朱颜改 问君能有几多愁恰似一江春水向东流",
        "style": "婉约"
    },
    {
        "title": "相见欢",
        "author": "李煜",
        "dynasty": "五代",
        "content": "无言独上西楼月如钩寂寞梧桐深院锁清秋 剪不断理还乱是离愁别是一般滋味在心头",
        "style": "婉约"
    },
    {
        "title": "浪淘沙令",
        "author": "李煜",
        "dynasty": "五代",
        "content": "帘外雨潺潺春意阑珊罗衾不耐五更寒 梦里不知身是客一晌贪欢 独自莫凭栏无限江山别时容易见时难 流水落花春去也天上人间",
        "style": "婉约"
    },
    {
        "title": "蝶恋花",
        "author": "欧阳修",
        "dynasty": "宋",
        "content": "庭院深深深几许杨柳堆烟帘幕无重数 玉勒雕鞍游冶处楼高不见章台路 雨横风狂三月暮门掩黄昏无计留春住 泪眼问花花不语乱红飞过秋千去",
        "style": "婉约"
    },
    {
        "title": "生查子·元夕",
        "author": "欧阳修",
        "dynasty": "宋",
        "content": "去年元夜时花市灯如昼 月上柳梢头人约黄昏后 今年元夜时月与灯依旧 不见去年人泪湿春衫袖",
        "style": "婉约"
    },
    {
        "title": "临江仙",
        "author": "晏几道",
        "dynasty": "宋",
        "content": "梦后楼台高锁酒醒帘幕低垂 去年春恨却来时落花人独立微雨燕双飞 记得小苹初见两重心字罗衣 琵琶弦上说相思当时明月在曾照彩云归",
        "style": "婉约"
    },
    {
        "title": "鹧鸪天",
        "author": "晏几道",
        "dynasty": "宋",
        "content": "彩袖殷勤捧玉钟当年拚却醉颜红 舞低杨柳楼心月歌尽桃花扇底风 从别后忆相逢几回魂梦与君同 今宵剩把银釭照犹恐相逢是梦中",
        "style": "婉约"
    },
    {
        "title": "青玉案",
        "author": "贺铸",
        "dynasty": "宋",
        "content": "凌波不过横塘路但目送芳尘去 锦瑟华年谁与度月桥花院琐窗朱户只有春知处 飞云冉冉蘅皋暮彩笔新题断肠句 试问闲愁都几许一川烟草满城风絮梅子黄时雨",
        "style": "婉约"
    },
    {
        "title": "玉楼春",
        "author": "晏殊",
        "dynasty": "宋",
        "content": "绿杨芳草长亭路年少抛人容易去 楼头残梦五更钟花底离愁三月雨 无情不似多情苦一寸还成千万缕 天涯地角有穷时只有相思无尽处",
        "style": "婉约"
    },
    {
        "title": "苏幕遮",
        "author": "范仲淹",
        "dynasty": "宋",
        "content": "碧云天黄叶地秋色连波波上寒烟翠 山映斜阳天接水芳草无情更在斜阳外 黯乡魂追旅思夜夜除非好梦留人睡 明月楼高休独倚酒入愁肠化作相思泪",
        "style": "婉约"
    },
    {
        "title": "蝶恋花",
        "author": "苏轼",
        "dynasty": "宋",
        "content": "花褪残红青杏小燕子飞时绿水人家绕 枝上柳绵吹又少天涯何处无芳草 墙里秋千墙外道墙外行人墙里佳人笑 笑渐不闻声渐悄多情却被无情恼",
        "style": "婉约"
    },
    {
        "title": "点绛唇",
        "author": "李清照",
        "dynasty": "宋",
        "content": "蹴罢秋千起来慵整纤纤手 露浓花瘦薄汗轻衣透 见客入来袜刬金钗溜 和羞走倚门回首却把青梅嗅",
        "style": "婉约"
    },
    {
        "title": "浣溪沙",
        "author": "李清照",
        "dynasty": "宋",
        "content": "绣面芙蓉一笑开斜飞宝鸭衬香腮眼波才动被人猜 一面风情深有韵半笺娇恨寄幽怀月移花影约重来",
        "style": "婉约"
    },
    {
        "title": "菩萨蛮",
        "author": "韦庄",
        "dynasty": "唐",
        "content": "人人尽说江南好游人只合江南老 春水碧于天画船听雨眠 垆边人似月皓腕凝霜雪 未老莫还乡还乡须断肠",
        "style": "婉约"
    },
    {
        "title": "女冠子",
        "author": "韦庄",
        "dynasty": "唐",
        "content": "四月十七正是去年今日别君时 忍泪佯低面含羞半敛眉 不知魂已断空有梦相随 除却天边月没人知",
        "style": "婉约"
    },
    {
        "title": "谒金门",
        "author": "冯延巳",
        "dynasty": "五代",
        "content": "风乍起吹皱一池春水 闲引鸳鸯香径里手挼红杏蕊 斗鸭阑干独倚碧玉搔头斜坠 终日望君君不至举头闻鹊喜",
        "style": "婉约"
    },
    {
        "title": "鹊踏枝",
        "author": "冯延巳",
        "dynasty": "五代",
        "content": "谁道闲情抛掷久每到春来惆怅还依旧 日日花前常病酒不辞镜里朱颜瘦 河畔青芜堤上柳为问新愁何事年年有 独立小桥风满袖平林新月人归后",
        "style": "婉约"
    },
    {
        "title": "摊破浣溪沙",
        "author": "李璟",
        "dynasty": "五代",
        "content": "菡萏香销翠叶残西风愁起绿波间 还与韶光共憔悴不堪看 细雨梦回鸡塞远小楼吹彻玉笙寒 多少泪珠何限恨倚阑干",
        "style": "婉约"
    },
    {
        "title": "相见欢",
        "author": "李煜",
        "dynasty": "五代",
        "content": "林花谢了春红太匆匆无奈朝来寒雨晚来风 胭脂泪相留醉几时重自是人生长恨水长东",
        "style": "婉约"
    },
    {
        "title": "清平乐",
        "author": "李煜",
        "dynasty": "五代",
        "content": "别来春半触目柔肠断 砌下落梅如雪乱拂了一身还满 雁来音信无凭路遥归梦难成 离恨恰如春草更行更远还生",
        "style": "婉约"
    },
    {
        "title": "乌夜啼",
        "author": "李煜",
        "dynasty": "五代",
        "content": "昨夜风兼雨帘帏飒飒秋声 烛残漏断频欹枕起坐不能平 世事漫随流水算来一梦浮生 醉乡路稳宜频到此外不堪行",
        "style": "婉约"
    },
    {
        "title": "浣溪沙",
        "author": "晏殊",
        "dynasty": "宋",
        "content": "一向年光有限身等闲离别易销魂酒筵歌席莫辞频 满目山河空念远落花风雨更伤春不如怜取眼前人",
        "style": "婉约"
    },
    {
        "title": "清平乐",
        "author": "晏殊",
        "dynasty": "宋",
        "content": "红笺小字说尽平生意鸿雁在云鱼在水惆怅此情难寄 斜阳独倚西楼遥山恰对帘钩人面不知何处绿波依旧东流",
        "style": "婉约"
    },
    {
        "title": "踏莎行",
        "author": "晏殊",
        "dynasty": "宋",
        "content": "小径红稀芳郊绿遍高台树色阴阴见 春风不解禁杨花蒙蒙乱扑行人面 翠叶藏莺朱帘隔燕炉香静逐游丝转 一场愁梦酒醒时斜阳却照深深院",
        "style": "婉约"
    },
    {
        "title": "八声甘州",
        "author": "柳永",
        "dynasty": "宋",
        "content": "对潇潇暮雨洒江天一番洗清秋 渐霜风凄紧关河冷落残照当楼 是处红衰翠减苒苒物华休 惟有长江水无语东流",
        "style": "婉约"
    },
    {
        "title": "忆帝京",
        "author": "柳永",
        "dynasty": "宋",
        "content": "薄衾小枕凉天气乍觉别离滋味 展转数寒更起了还重睡 毕竟不成眠一夜长如岁 也拟待却回征辔又争奈已行千里",
        "style": "婉约"
    },
    {
        "title": "思远人",
        "author": "晏几道",
        "dynasty": "宋",
        "content": "红叶黄花秋意晚千里念行客 飞云过尽归鸿无信何处寄书得 泪弹不尽临窗滴就砚旋研墨 渐写到别来此情深处红笺为无色",
        "style": "婉约"
    },
    {
        "title": "惜分飞",
        "author": "毛滂",
        "dynasty": "宋",
        "content": "泪湿阑干花著露愁到眉峰碧聚 此恨平分取更无言语空相觑 短雨残云无意绪寂寞朝朝暮暮 今夜山深处断魂分付潮回去",
        "style": "婉约"
    },
    {
        "title": "临江仙",
        "author": "陈与义",
        "dynasty": "宋",
        "content": "忆昔午桥桥上饮坐中多是豪英 长沟流月去无声杏花疏影里吹笛到天明 二十余年如一梦此身虽在堪惊 闲登小阁看新晴古今多少事渔唱起三更",
        "style": "婉约"
    },
    {
        "title": "唐多令",
        "author": "吴文英",
        "dynasty": "宋",
        "content": "何处合成愁离人心上秋 纵芭蕉不雨也飕飕 都道晚凉天气好有明月怕登楼 年事梦中消花空烟水流 燕辞归客尚淹留 垂柳不萦裙带住漫长是系行舟",
        "style": "婉约"
    },
    {
        "title": "风入松",
        "author": "吴文英",
        "dynasty": "宋",
        "content": "听风听雨过清明愁草瘗花铭 楼前绿暗分携路一丝柳一寸柔情 料峭春寒中酒交加晓梦啼莺 西园日日扫林亭依旧赏新晴 黄蜂频扑秋千索有当时纤手香凝 惆怅双鸳不到幽阶一夜苔生",
        "style": "婉约"
    },
    {
        "title": "扬州慢",
        "author": "姜夔",
        "dynasty": "宋",
        "content": "淮左名都竹西佳处解鞍少驻初程 过春风十里尽荠麦青青 自胡马窥江去后废池乔木犹厌言兵 渐黄昏清角吹寒都在空城",
        "style": "婉约"
    },
    {
        "title": "暗香",
        "author": "姜夔",
        "dynasty": "宋",
        "content": "旧时月色算几番照我梅边吹笛 唤起玉人不管清寒与攀摘 何逊而今渐老都忘却春风词笔 但怪得竹外疏花香冷入瑶席",
        "style": "婉约"
    },
    {
        "title": "疏影",
        "author": "姜夔",
        "dynasty": "宋",
        "content": "苔枝缀玉有翠禽小小枝上同宿 客里相逢篱角黄昏无言自倚修竹 昭君不惯胡沙远但暗忆江南江北 想佩环月夜归来化作此花幽独",
        "style": "婉约"
    },
    {
        "title": "满庭芳",
        "author": "秦观",
        "dynasty": "宋",
        "content": "山抹微云天粘衰草画角声断谯门 暂停征棹聊共引离尊 多少蓬莱旧事空回首烟霭纷纷 斜阳外寒鸦万点流水绕孤村",
        "style": "婉约"
    },
    {
        "title": "望江南",
        "author": "温庭筠",
        "dynasty": "唐",
        "content": "千万恨恨极在天涯 山月不知心里事水风空落眼前花 摇曳碧云斜",
        "style": "婉约"
    },
    {
        "title": "归园田居·其三",
        "author": "陶渊明",
        "dynasty": "晋",
        "content": "种豆南山下草盛豆苗稀 晨兴理荒秽带月荷锄归 道狭草木长夕露沾我衣 衣沾不足惜但使愿无违",
        "style": "田园"
    },
    {
        "title": "饮酒·其五",
        "author": "陶渊明",
        "dynasty": "晋",
        "content": "结庐在人境而无车马喧 问君何能尔心远地自偏 采菊东篱下悠然见南山 山气日夕佳飞鸟相与还 此中有真意欲辨已忘言",
        "style": "田园"
    },
    {
        "title": "饮酒·其七",
        "author": "陶渊明",
        "dynasty": "晋",
        "content": "秋菊有佳色裛露掇其英 泛此忘忧物远我遗世情 一觞虽独进杯尽壶自倾 日入群动息归鸟趋林鸣",
        "style": "田园"
    },
    {
        "title": "读山海经",
        "author": "陶渊明",
        "dynasty": "晋",
        "content": "孟夏草木长绕屋树扶疏 众鸟欣有托吾亦爱吾庐 既耕亦已种时还读我书 穷巷隔深辙颇回故人车",
        "style": "田园"
    },
    {
        "title": "桃花源记",
        "author": "陶渊明",
        "dynasty": "晋",
        "content": "晋太元中武陵人捕鱼为业 缘溪行忘路之远近 忽逢桃花林夹岸数百步 中无杂树芳草鲜美落英缤纷",
        "style": "田园"
    },
    {
        "title": "山居秋暝",
        "author": "王维",
        "dynasty": "唐",
        "content": "空山新雨后 天气晚来秋 明月松间照 清泉石上流 竹喧归浣女 莲动下渔舟 随意春芳歇 王孙自可留",
        "style": "田园"
    },
    {
        "title": "鸟鸣涧",
        "author": "王维",
        "dynasty": "唐",
        "content": "人闲桂花落 夜静春山空 月出惊山鸟 时鸣春涧中",
        "style": "田园"
    },
    {
        "title": "鹿柴",
        "author": "王维",
        "dynasty": "唐",
        "content": "空山不见人 但闻人语响 返景入深林 复照青苔上",
        "style": "田园"
    },
    {
        "title": "竹里馆",
        "author": "王维",
        "dynasty": "唐",
        "content": "独坐幽篁里 弹琴复长啸 深林人不知 明月来相照",
        "style": "田园"
    },
    {
        "title": "送元二使安西",
        "author": "王维",
        "dynasty": "唐",
        "content": "渭城朝雨浥轻尘 客舍青青柳色新 劝君更尽一杯酒 西出阳关无故人",
        "style": "田园"
    },
    {
        "title": "终南别业",
        "author": "王维",
        "dynasty": "唐",
        "content": "中岁颇好道晚家南山陲 兴来每独往胜事空自知 行到水穷处坐看云起时 偶然值林叟谈笑无还期",
        "style": "田园"
    },
    {
        "title": "青溪",
        "author": "王维",
        "dynasty": "唐",
        "content": "言入黄花川每逐青溪水 随山将万转趣途无百里 声喧乱石中色静深松里 漾漾泛菱荇澄澄映葭苇",
        "style": "田园"
    },
    {
        "title": "渭川田家",
        "author": "王维",
        "dynasty": "唐",
        "content": "斜阳照墟落穷巷牛羊归 野老念牧童倚杖候荆扉 雉雊麦苗秀蚕眠桑叶稀 田夫荷锄至相见语依依",
        "style": "田园"
    },
    {
        "title": "春晓",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "春眠不觉晓 处处闻啼鸟 夜来风雨声 花落知多少",
        "style": "田园"
    },
    {
        "title": "过故人庄",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "故人具鸡黍邀我至田家 绿树村边合青山郭外斜 开轩面场圃把酒话桑麻 待到重阳日还来就菊花",
        "style": "田园"
    },
    {
        "title": "宿建德江",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "移舟泊烟渚 日暮客愁新 野旷天低树 江清月近人",
        "style": "田园"
    },
    {
        "title": "临洞庭湖赠张丞相",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "八月湖水平 涵虚混太清 气蒸云梦泽 波撼岳阳城 欲济无舟楫 端居耻圣明 坐观垂钓者 徒有羡鱼情",
        "style": "田园"
    },
    {
        "title": "夏日南亭怀辛大",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "山光忽西落池月渐东上 散发乘夕凉开轩卧闲敞 荷风送香气竹露滴清响 欲取鸣琴弹恨无知音赏",
        "style": "田园"
    },
    {
        "title": "秋登兰山寄张五",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "北山白云里隐者自怡悦 相望试登高心随雁飞灭 愁因薄暮起兴是清秋发 时见归村人沙行渡头歇",
        "style": "田园"
    },
    {
        "title": "滁州西涧",
        "author": "韦应物",
        "dynasty": "唐",
        "content": "独怜幽草涧边生 上有黄鹂深树鸣 春潮带雨晚来急 野渡无人舟自横",
        "style": "田园"
    },
    {
        "title": "辋川闲居赠裴秀才迪",
        "author": "王维",
        "dynasty": "唐",
        "content": "寒山转苍翠秋水日潺湲 倚杖柴门外临风听暮蝉 渡头馀落日墟里上孤烟 复值接舆醉狂歌五柳前",
        "style": "田园"
    },
    {
        "title": "积雨辋川庄作",
        "author": "王维",
        "dynasty": "唐",
        "content": "积雨空林烟火迟蒸藜炊黍饷东菑 漠漠水田飞白鹭阴阴夏木啭黄鹂 山中习静观朝槿松下清斋折露葵 野老与人争席罢海鸥何事更相疑",
        "style": "田园"
    },
    {
        "title": "春中田园作",
        "author": "王维",
        "dynasty": "唐",
        "content": "屋上春鸠鸣村边杏花白 持斧伐远扬荷锄觇泉脉 归燕识故巢旧人看新历 临觞忽不御惆怅远行客",
        "style": "田园"
    },
    {
        "title": "宿王昌龄隐居",
        "author": "常建",
        "dynasty": "唐",
        "content": "清溪深不测隐处唯孤云 松际露微月清光犹为君 茅亭宿花影药院滋苔纹 余亦谢时去西山鸾鹤群",
        "style": "田园"
    },
    {
        "title": "题破山寺后禅院",
        "author": "常建",
        "dynasty": "唐",
        "content": "清晨入古寺初日照高林 曲径通幽处禅房花木深 山光悦鸟性潭影空人心 万籁此俱寂但馀钟磬音",
        "style": "田园"
    },
    {
        "title": "商山早行",
        "author": "温庭筠",
        "dynasty": "唐",
        "content": "晨起动征铎客行悲故乡 鸡声茅店月人迹板桥霜 槲叶落山路枳花明驿墙 因思杜陵梦凫雁满回塘",
        "style": "田园"
    },
    {
        "title": "江村即事",
        "author": "司空曙",
        "dynasty": "唐",
        "content": "钓罢归来不系船江村月落正堪眠 纵然一夜风吹去只在芦花浅水边",
        "style": "田园"
    },
    {
        "title": "雨过山村",
        "author": "王建",
        "dynasty": "唐",
        "content": "雨里鸡鸣一两家竹溪村路板桥斜 妇姑相唤浴蚕去闲着中庭栀子花",
        "style": "田园"
    },
    {
        "title": "社日",
        "author": "王驾",
        "dynasty": "唐",
        "content": "鹅湖山下稻粱肥豚栅鸡栖半掩扉 桑柘影斜春社散家家扶得醉人归",
        "style": "田园"
    },
    {
        "title": "四时田园杂兴·其一",
        "author": "范成大",
        "dynasty": "宋",
        "content": "昼出耘田夜绩麻村庄儿女各当家 童孙未解供耕织也傍桑阴学种瓜",
        "style": "田园"
    },
    {
        "title": "四时田园杂兴·其二",
        "author": "范成大",
        "dynasty": "宋",
        "content": "梅子金黄杏子肥麦花雪白菜花稀 日长篱落无人过惟有蜻蜓蛱蝶飞",
        "style": "田园"
    },
    {
        "title": "村居",
        "author": "高鼎",
        "dynasty": "清",
        "content": "草长莺飞二月天拂堤杨柳醉春烟 儿童散学归来早忙趁东风放纸鸢",
        "style": "田园"
    },
    {
        "title": "小池",
        "author": "杨万里",
        "dynasty": "宋",
        "content": "泉眼无声惜细流树阴照水爱晴柔 小荷才露尖尖角早有蜻蜓立上头",
        "style": "田园"
    },
    {
        "title": "晓出净慈寺送林子方",
        "author": "杨万里",
        "dynasty": "宋",
        "content": "毕竟西湖六月中风光不与四时同 接天莲叶无穷碧映日荷花别样红",
        "style": "田园"
    },
    {
        "title": "宿新市徐公店",
        "author": "杨万里",
        "dynasty": "宋",
        "content": "篱落疏疏一径深树头新绿未成阴 儿童急走追黄蝶飞入菜花无处寻",
        "style": "田园"
    },
    {
        "title": "清平乐·村居",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "茅檐低小溪上青青草 醉里吴音相媚好白发谁家翁媪 大儿锄豆溪东中儿正织鸡笼 最喜小儿亡赖溪头卧剥莲蓬",
        "style": "田园"
    },
    {
        "title": "西江月·夜行黄沙道中",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "明月别枝惊鹊清风半夜鸣蝉 稻花香里说丰年听取蛙声一片 七八个星天外两三点雨山前 旧时茅店社林边路转溪桥忽见",
        "style": "田园"
    },
    {
        "title": "大林寺桃花",
        "author": "白居易",
        "dynasty": "唐",
        "content": "人间四月芳菲尽 山寺桃花始盛开 长恨春归无觅处 不知转入此中来",
        "style": "田园"
    },
    {
        "title": "村夜",
        "author": "白居易",
        "dynasty": "唐",
        "content": "霜草苍苍虫切切村南村北行人绝 独出门前望野田月明荞麦花如雪",
        "style": "田园"
    },
    {
        "title": "问刘十九",
        "author": "白居易",
        "dynasty": "唐",
        "content": "绿蚁新醅酒红泥小火炉 晚来天欲雪能饮一杯无",
        "style": "田园"
    },
    {
        "title": "江雪",
        "author": "柳宗元",
        "dynasty": "唐",
        "content": "千山鸟飞绝 万径人踪灭 孤舟蓑笠翁 独钓寒江雪",
        "style": "田园"
    },
    {
        "title": "渔翁",
        "author": "柳宗元",
        "dynasty": "唐",
        "content": "渔翁夜傍西岩宿晓汲清湘燃楚竹 烟销日出不见人欸乃一声山水绿 回看天际下中流岩上无心云相逐",
        "style": "田园"
    },
    {
        "title": "溪居",
        "author": "柳宗元",
        "dynasty": "唐",
        "content": "久为簪组累幸此南夷谪 闲依农圃邻偶似山林客 晓耕翻露草夜榜响溪石 来往不逢人长歌楚天碧",
        "style": "田园"
    },
    {
        "title": "山中",
        "author": "王维",
        "dynasty": "唐",
        "content": "荆溪白石出天寒红叶稀 山路元无雨空翠湿人衣",
        "style": "田园"
    },
    {
        "title": "书湖阴先生壁",
        "author": "王安石",
        "dynasty": "宋",
        "content": "茅檐长扫净无苔花木成畦手自栽 一水护田将绿绕两山排闼送青来",
        "style": "田园"
    },
    {
        "title": "泊船瓜洲",
        "author": "王安石",
        "dynasty": "宋",
        "content": "京口瓜洲一水间钟山只隔数重山 春风又绿江南岸明月何时照我还",
        "style": "田园"
    },
    {
        "title": "约客",
        "author": "赵师秀",
        "dynasty": "宋",
        "content": "黄梅时节家家雨青草池塘处处蛙 有约不来过夜半闲敲棋子落灯花",
        "style": "田园"
    },
    {
        "title": "游园不值",
        "author": "叶绍翁",
        "dynasty": "宋",
        "content": "应怜屐齿印苍苔小扣柴扉久不开 春色满园关不住一枝红杏出墙来",
        "style": "田园"
    },
    {
        "title": "乡村四月",
        "author": "翁卷",
        "dynasty": "宋",
        "content": "绿遍山原白满川子规声里雨如烟 乡村四月闲人少才了蚕桑又插田",
        "style": "田园"
    },
    {
        "title": "三衢道中",
        "author": "曾几",
        "dynasty": "宋",
        "content": "梅子黄时日日晴小溪泛尽却山行 绿阴不减来时路添得黄鹂四五声",
        "style": "田园"
    },
    {
        "title": "绝句",
        "author": "杜甫",
        "dynasty": "唐",
        "content": "两个黄鹂鸣翠柳一行白鹭上青天 窗含西岭千秋雪门泊东吴万里船",
        "style": "田园"
    },
    {
        "title": "绝句",
        "author": "杜甫",
        "dynasty": "唐",
        "content": "迟日江山丽春风花草香 泥融飞燕子沙暖睡鸳鸯",
        "style": "田园"
    },
    {
        "title": "江畔独步寻花",
        "author": "杜甫",
        "dynasty": "唐",
        "content": "黄四娘家花满蹊千朵万朵压枝低 留连戏蝶时时舞自在娇莺恰恰啼",
        "style": "田园"
    },
    {
        "title": "春夜喜雨",
        "author": "杜甫",
        "dynasty": "唐",
        "content": "好雨知时节当春乃发生 随风潜入夜润物细无声 野径云俱黑江船火独明 晓看红湿处花重锦官城",
        "style": "田园"
    },
    {
        "title": "逢雪宿芙蓉山主人",
        "author": "刘长卿",
        "dynasty": "唐",
        "content": "日暮苍山远天寒白屋贫 柴门闻犬吠风雪夜归人",
        "style": "田园"
    },
    {
        "title": "寻南溪常山道人隐居",
        "author": "刘长卿",
        "dynasty": "唐",
        "content": "一路经行处莓苔见履痕 白云依静渚春草闭闲门 过雨看松色随山到水源 溪花与禅意相对亦忘言",
        "style": "田园"
    },
    {
        "title": "夜归鹿门歌",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "山寺钟鸣昼已昏渔梁渡头争渡喧 人随沙岸向江村余亦乘舟归鹿门 鹿门月照开烟树忽到庞公栖隐处 岩扉松径长寂寥惟有幽人自来去",
        "style": "田园"
    },
    {
        "title": "早寒有怀",
        "author": "孟浩然",
        "dynasty": "唐",
        "content": "木落雁南度北风江上寒 我家襄水曲遥隔楚云端 乡泪客中尽孤帆天际看 迷津欲有问平海夕漫漫",
        "style": "田园"
    },
    {
        "title": "新晴野望",
        "author": "王维",
        "dynasty": "唐",
        "content": "新晴原野旷极目无氛垢 郭门临渡头村树连溪口 白水明田外碧峰出山后 农月无闲人倾家事南亩",
        "style": "田园"
    },
    {
        "title": "秋日赴阙题潼关驿楼",
        "author": "许浑",
        "dynasty": "唐",
        "content": "红叶晚萧萧长亭酒一瓢 残云归太华疏雨过中条 树色随山迥河声入海遥 帝乡明日到犹自梦渔樵",
        "style": "田园"
    },
    {
        "title": "田园乐",
        "author": "王维",
        "dynasty": "唐",
        "content": "桃红复含宿雨柳绿更带朝烟 花落家童未扫莺啼山客犹眠",
        "style": "田园"
    },
    {
        "title": "鹧鸪天·代人赋",
        "author": "辛弃疾",
        "dynasty": "宋",
        "content": "晚日寒鸦一片愁柳塘新绿却温柔 若教眼底无离恨不信人间有白头 肠已断泪难收相思重上小红楼 情知已被山遮断频倚阑干不自由",
        "style": "田园"
    },
    {
        "title": "八六子",
        "author": "秦观",
        "dynasty": "宋",
        "content": "倚危亭恨如芳草萋萋刬尽还生 念柳外青骢别后水边红袂分时怆然暗惊 无端天与娉婷夜月一帘幽梦春风十里柔情",
        "style": "婉约"
    },
    {
        "title": "千秋岁",
        "author": "秦观",
        "dynasty": "宋",
        "content": "水边沙外城郭春寒退 花影乱莺声碎 飘零疏酒盏离别宽衣带 人不见碧云暮合空相对",
        "style": "婉约"
    },
    {
        "title": "留春令",
        "author": "晏几道",
        "dynasty": "宋",
        "content": "画屏天畔梦回依约十洲云水 手捻红笺寄人书写无限伤春事 别浦高楼曾漫倚对江南千里 楼下分流水声中有当日凭高泪",
        "style": "婉约"
    },
    {
        "title": "秋蕊香",
        "author": "晏几道",
        "dynasty": "宋",
        "content": "池苑清阴欲就还傍送春时候 眼中人去难欢偶谁共一杯芳酒 朱阑碧砌皆如旧记携手 有情不管别离久情在相逢终有",
        "style": "婉约"
    },
    {
        "title": "绮罗香",
        "author": "史达祖",
        "dynasty": "宋",
        "content": "做冷欺花将烟困柳千里偷催春暮 尽日冥迷愁里欲飞还住 惊粉重蝶宿西园喜泥润燕归南浦 最妨他佳约风流钿车不到杜陵路",
        "style": "婉约"
    },
    {
        "title": "双双燕",
        "author": "史达祖",
        "dynasty": "宋",
        "content": "过春社了度帘幕中间去年尘冷 差池欲住试入旧巢相并 还相雕梁藻井又软语商量不定 飘然快拂花梢翠尾分开红影",
        "style": "婉约"
    },
    {
        "title": "高阳台",
        "author": "吴文英",
        "dynasty": "宋",
        "content": "修竹凝妆垂杨系马凭阑浅画成图 山色谁题楼前有雁斜书 东风紧送斜阳下弄旧寒晚酒醒馀 自消凝能几花前顿老相如",
        "style": "婉约"
    },
    {
        "title": "夜半乐",
        "author": "柳永",
        "dynasty": "宋",
        "content": "冻云黯淡天气扁舟一叶乘兴离江渚 渡万壑千岩越溪深处怒涛渐息樵风乍起 更闻商旅相呼片帆高举泛画鹢翩翩过南浦",
        "style": "婉约"
    },
    {
        "title": "玉蝴蝶",
        "author": "柳永",
        "dynasty": "宋",
        "content": "望处雨收云断凭阑悄悄目送秋光 晚景萧疏堪动宋玉悲凉 水风轻苹花渐老月露冷梧叶飘黄 遣情伤故人何在烟水茫茫",
        "style": "婉约"
    },
    {
        "title": "绿罗裙",
        "author": "贺铸",
        "dynasty": "宋",
        "content": "东风柳陌长闭月花房小 应念画眉人拂镜啼新晓 伤心南浦波回首青门道 记得绿罗裙处处怜芳草",
        "style": "婉约"
    },
    {
        "title": "望江南",
        "author": "温庭筠",
        "dynasty": "唐",
        "content": "梳洗罢独倚望江楼 过尽千帆皆不是斜晖脉脉水悠悠 肠断白苹洲",
        "style": "婉约"
    },
    {
        "title": "八六子",
        "author": "秦观",
        "dynasty": "宋",
        "content": "倚危亭恨如芳草萋萋刬尽还生 念柳外青骢别后水边红袂分时怆然暗惊 无端天与娉婷夜月一帘幽梦春风十里柔情",
        "style": "婉约"
    },
    {
        "title": "千秋岁",
        "author": "秦观",
        "dynasty": "宋",
        "content": "水边沙外城郭春寒退 花影乱莺声碎 飘零疏酒盏离别宽衣带 人不见碧云暮合空相对",
        "style": "婉约"
    },
    {
        "title": "留春令",
        "author": "晏几道",
        "dynasty": "宋",
        "content": "画屏天畔梦回依约十洲云水 手捻红笺寄人书写无限伤春事 别浦高楼曾漫倚对江南千里 楼下分流水声中有当日凭高泪",
        "style": "婉约"
    },
    {
        "title": "秋蕊香",
        "author": "晏几道",
        "dynasty": "宋",
        "content": "池苑清阴欲就还傍送春时候 眼中人去难欢偶谁共一杯芳酒 朱阑碧砌皆如旧记携手 有情不管别离久情在相逢终有",
        "style": "婉约"
    },
    {
        "title": "菩萨蛮",
        "author": "陈克",
        "dynasty": "宋",
        "content": "赤栏桥近香街直笼街细柳娇无力 金碧上青空花晴帘影红 黄衫飞白马日日青楼下 醉眼不逢人午香吹暗尘",
        "style": "婉约"
    },
    {
        "title": "绮罗香·咏春雨",
        "author": "史达祖",
        "dynasty": "宋",
        "content": "做冷欺花将烟困柳千里偷催春暮 尽日冥迷愁里欲飞还住 惊粉重蝶宿西园喜泥润燕归南浦 最妨他佳约风流钿车不到杜陵路",
        "style": "婉约"
    },
    {
        "title": "双双燕",
        "author": "史达祖",
        "dynasty": "宋",
        "content": "过春社了度帘幕中间去年尘冷 差池欲住试入旧巢相并 还相雕梁藻井又软语商量不定 飘然快拂花梢翠尾分开红影",
        "style": "婉约"
    },
    {
        "title": "高阳台",
        "author": "吴文英",
        "dynasty": "宋",
        "content": "修竹凝妆垂杨系马凭阑浅画成图 山色谁题楼前有雁斜书 东风紧送斜阳下弄旧寒晚酒醒馀 自消凝能几花前顿老相如",
        "style": "婉约"
    },
    {
        "title": "夜半乐",
        "author": "柳永",
        "dynasty": "宋",
        "content": "冻云黯淡天气扁舟一叶乘兴离江渚 渡万壑千岩越溪深处怒涛渐息樵风乍起 更闻商旅相呼片帆高举泛画鹢翩翩过南浦",
        "style": "婉约"
    },
    {
        "title": "浣溪沙",
        "author": "秦观",
        "dynasty": "宋",
        "content": "漠漠轻阴晚自开青天白日映楼台 曲江水满花千树有底忙时不肯来",
        "style": "婉约"
    },
    {
        "title": "酒泉子",
        "author": "温庭筠",
        "dynasty": "唐",
        "content": "楚女不归楼枕小河春水 月孤明风又起杏花稀",
        "style": "婉约"
    },
    {
        "title": "玉蝴蝶",
        "author": "柳永",
        "dynasty": "宋",
        "content": "望处雨收云断凭阑悄悄目送秋光 晚景萧疏堪动宋玉悲凉 水风轻蘋花渐老月露冷梧叶飘黄 遣情伤故人何在烟水茫茫",
        "style": "婉约"
    },
    {
        "title": "夜合花",
        "author": "晁补之",
        "dynasty": "宋",
        "content": "百紫千红占春多少共推绝世花王 西都万家俱好不为姚黄 谩肠断巫阳对沉香亭北新妆 记清平调词成进天一醉君王",
        "style": "婉约"
    },
    {
        "title": "绿罗裙",
        "author": "贺铸",
        "dynasty": "宋",
        "content": "东风柳陌长闭月花房小 应念画眉人拂镜啼新晓 伤心南浦波回首青门道 记得绿罗裙处处怜芳草",
        "style": "婉约"
    },
    {
        "title": "秣陵道中口占",
        "author": "王安石",
        "dynasty": "宋",
        "content": "经世才难就田园路欲迷 殷勤将白发下马照青溪",
        "style": "田园"
    },
    {
        "title": "秋思",
        "author": "张籍",
        "dynasty": "唐",
        "content": "洛阳城里见秋风欲作家书意万重 复恐匆匆说不尽行人临发又开封",
        "style": "田园"
    },
    {
        "title": "望月怀远",
        "author": "张九龄",
        "dynasty": "唐",
        "content": "海上生明月天涯共此时 情人怨遥夜竟夕起相思 灭烛怜光满披衣觉露滋 不堪盈手赠还寝梦佳期",
        "style": "田园"
    },
    {
        "title": "野望",
        "author": "王绩",
        "dynasty": "唐",
        "content": "东皋薄暮望徙倚欲何依 树树皆秋色山山唯落晖 牧人驱犊返猎马带禽归 相顾无相识长歌怀采薇",
        "style": "田园"
    },
    {
        "title": "山中",
        "author": "王勃",
        "dynasty": "唐",
        "content": "长江悲已滞万里念将归 况属高风晚山山黄叶飞",
        "style": "田园"
    },
    {
        "title": "春庄",
        "author": "王勃",
        "dynasty": "唐",
        "content": "山中兰叶径城外李桃园 岂知人事静不觉鸟声喧",
        "style": "田园"
    },
    {
        "title": "田家三首",
        "author": "柳宗元",
        "dynasty": "唐",
        "content": "蓐食徇所务驱牛向东阡 鸡鸣村巷白夜色归暮田 札札耒耜声飞飞来乌鸢 竭兹筋力事持用穷岁年",
        "style": "田园"
    },
    {
        "title": "首春",
        "author": "李世民",
        "dynasty": "唐",
        "content": "寒随穷律变春逐鸟声开 初风飘带柳晚雪间花梅",
        "style": "田园"
    },
    {
        "title": "咏柳",
        "author": "贺知章",
        "dynasty": "唐",
        "content": "碧玉妆成一树高万条垂下绿丝绦 不知细叶谁裁出二月春风似剪刀",
        "style": "田园"
    },
    {
        "title": "回乡偶书",
        "author": "贺知章",
        "dynasty": "唐",
        "content": "少小离家老大回乡音无改鬓毛衰 儿童相见不相识笑问客从何处来",
        "style": "田园"
    },
    {
        "title": "阙题",
        "author": "刘昚虚",
        "dynasty": "唐",
        "content": "道由白云尽春与青溪长 时有落花至远随流水香 闲门向山路深柳读书堂 幽映每白日清辉照衣裳",
        "style": "田园"
    },
    {
        "title": "北山",
        "author": "王安石",
        "dynasty": "宋",
        "content": "北山输绿涨横陂直堑回塘滟滟时 细数落花因坐久缓寻芳草得归迟",
        "style": "田园"
    },
    {
        "title": "江村晚眺",
        "author": "戴复古",
        "dynasty": "宋",
        "content": "江头落日照平沙潮退渔船阁岸斜 白鸟一双临水立见人惊起入芦花",
        "style": "田园"
    },
    {
        "title": "村晚",
        "author": "雷震",
        "dynasty": "宋",
        "content": "草满池塘水满陂山衔落日浸寒漪 牧童归去横牛背短笛无腔信口吹",
        "style": "田园"
    },
    {
        "title": "桑茶坑道中",
        "author": "杨万里",
        "dynasty": "宋",
        "content": "晴明风日雨干时草满花堤水满溪 童子柳阴眠正着一牛吃过柳阴西",
        "style": "田园"
    },
    {
        "title": "池州翠微亭",
        "author": "岳飞",
        "dynasty": "宋",
        "content": "经年尘土满征衣特特寻芳上翠微 好水好山看不足马蹄催趁月明归",
        "style": "田园"
    },
    {
        "title": "天平山中",
        "author": "杨基",
        "dynasty": "明",
        "content": "细雨茸茸湿楝花南风树树熟枇杷 徐行不记山深浅一路莺啼送到家",
        "style": "田园"
    }
]

print(f'[poetry_land] Loaded {len(POEMS)} poems')

POSITIVE_WORDS = {"春","花","月","明","清","新","好","美","香","笑","歌","乐","爱","喜","欢","碧","翠","金","玉","红","霞","虹","祥","瑞","秀","丽","雅","幽","静","佳","妙","奇","灵","仙","圣","英","豪","壮","高","远","长","满","丰","盈","茂","荣","润","泽","甜","和","平","顺","吉","绚","灿","辉","照","暖","融","怡","恬","逸","悠","达","畅","蔚","苍","闲","醉","欣","悦","舒","安"}
NEGATIVE_WORDS = {"愁","恨","悲","哀","伤","痛","苦","泪","泣","哭","寒","冷","凉","冰","霜","雪","秋","暮","夕","晚","昏","暗","阴","残","败","落","谢","衰","枯","荒","凄","惨","寂","寞","孤","独","空","虚","茫","迷","惘","忧","虑","患","病","老","死","灭","亡","乱","战","怨","惊","恐","怖","怕","疲","倦","闷","怒","嫉","妒","碎","断","折","裂","破","毁","绝","寥","疏","萧","瑟","飘","零","离","散","别","阻"}

# ── Helper functions ──
def tokenize_text(text):
    return [w for w in jieba.cut(text) if len(w.strip()) > 0]

@st.cache_data
def train_markov(style, order=2):
    texts = [p["content"] for p in POEMS if p["style"] == style]
    mc = {}
    examples = []
    for t in texts[:10]:
        examples.append(t[:40])
    for t in texts:
        toks = tokenize_text(t)
        for i in range(len(toks) - order):
            key = str(toks[i:i+order])
            if key not in mc:
                mc[key] = []
            mc[key].append(toks[i+order] if i+order < len(toks) else "")
    return mc, examples

def markov_gen(mc, order=2, target_len=28):
    if not mc:
        return "（数据不足）"
    keys = list(mc.keys())
    for _ in range(200):
        k = _random.choice(keys)
        parts = [x for x in eval(k)]
        result = list(parts)
        for _ in range(100):
            key = str(result[-order:]) if len(result) >= order else k
            if len(result) < order:
                break
            nxt = _random.choice(mc.get(key, [""]))
            if not nxt:
                break
            result.append(nxt)
            if len("".join(result)) >= target_len:
                break
        text = "".join(result)
        if len(text) >= 20:
            lines = []
            i = 0
            for plen in [5,5,5,5]:
                if i+plen <= len(text):
                    lines.append(text[i:i+plen])
                    i += plen
            if len(lines) == 4:
                return "\n".join(lines)
    return text[:20]

def sentiment(text):
    pos = sum(1 for c in text if c in POSITIVE_WORDS)
    neg = sum(1 for c in text if c in NEGATIVE_WORDS)
    total = pos + neg
    score = (pos - neg) / total if total > 0 else 0
    return {"score": score, "pos": pos, "neg": neg, 
            "pos_words": list(set(c for c in text if c in POSITIVE_WORDS))[:15],
            "neg_words": list(set(c for c in text if c in NEGATIVE_WORDS))[:15]}

@st.cache_data
def download_data():
    urls = [
        "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.0.json",
        "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.1.json",
        "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/%E5%85%A8%E5%AE%8B%E8%AF%8D/poet.song.0.json",
    ]
    new = []
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                for item in data[:60]:
                    pars = " ".join(item.get("paragraphs", []))
                    if len(pars) > 10:
                        new.append({
                            "title": item.get("title", ""), "author": item.get("author", ""),
                            "dynasty": "唐" if "唐" in url else "宋",
                            "content": pars, "style": _random.choice(["豪放","婉约","田园"])
                        })
        except:
            pass
    return new



def format_poem(content):
    """Add Chinese punctuation to poem text for readability"""
    lines = content.strip().split()
    if not lines:
        return content
    punctuated = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i == len(lines) - 1:
            punctuated.append(line + "\u3002")
        elif i % 2 == 0:
            punctuated.append(line + "\uff0c")
        else:
            punctuated.append(line + "\u3002")
    return "".join(punctuated)

def get_stats(poems):
    df = pd.DataFrame(poems)
    t = len(df)
    a = df["author"].nunique()
    dc = df["dynasty"].value_counts().reset_index()
    dc.columns = ["朝代","数量"]
    sc = df["style"].value_counts().reset_index()
    sc.columns = ["风格","数量"]
    all_text = "".join(df["content"].tolist())
    words = [w for w in tokenize_text(all_text) if len(w) > 1]
    top10 = pd.DataFrame(Counter(words).most_common(10), columns=["字/词","出现次数"])
    return t, a, dc, sc, top10

def find_similar(query, poems, top_k=3):
    """Use TF-IDF vectorization to find similar poems"""
    texts = [p["content"] for p in poems]
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2,3), max_features=500)
    try:
        X = vec.fit_transform(texts)
        qv = vec.transform([query])
        scores = (X @ qv.T).toarray().flatten()
        indices = scores.argsort()[-top_k:][::-1]
        return [(poems[i], scores[i]) for i in indices if scores[i] > 0]
    except:
        return []


def format_poem(content):
    if not content:
        return content
    lines = [l.strip() for l in content.strip().split() if l.strip()]
    if not lines:
        return content
    result = []
    for i, line in enumerate(lines):
        if i == len(lines) - 1:
            result.append(line + "\u3002")
        elif i % 2 == 0:
            result.append(line + "\uff0c")
        else:
            result.append(line + "\u3002")
    return "".join(result)


def poet_words(poems, poet, n=10):
    texts = [p["content"] for p in poems if p["author"] == poet]
    words = [w for w in tokenize_text("".join(texts)) if len(w) > 1]
    return Counter(words).most_common(n)

# ════════════════════════════════════════
# Main UI
# ════════════════════════════════════════

st.sidebar.markdown("""<div style="text-align:center;padding:1rem 0.5rem;"><div style="font-family:Noto Serif SC,serif;font-size:1.6rem;font-weight:700;color:#3A2A1A;letter-spacing:0.08em;">poetry_land</div><div style="font-size:0.75rem;color:#8B7355;letter-spacing:0.1em;margin-top:0.2rem;">&#10022; &#35799;&#35789;&#20114;&#21160;&#25506;&#32034;&#24179;&#21488; &#10022;</div><div style="margin:0.5rem auto;width:80px;height:1px;background:linear-gradient(90deg,transparent,#C04040,transparent);"></div></div>""", unsafe_allow_html=True)
mode = st.sidebar.radio("导航", ["首页","飞花令","诗词生成","诗人对比","情感分析"], key="nav")
total, authors, dynasty_counts, style_counts, top_chars = get_stats(POEMS)

if mode == "飞花令":
    st.markdown("## 飞花令 · 诗词搜索")
    kw = st.sidebar.text_input("输入搜索字", value="月")
    if st.sidebar.button("搜索", key="fh_btn") or kw:
        results = []
        for p in POEMS:
            for line in p["content"].split():
                if kw in line:
                    results.append({"诗句":line,"诗名":p["title"],"作者":p["author"],"朝代":p["dynasty"]})
        st.success(f"找到 {len(results)} 句")
        if results:
            df = pd.DataFrame(results)
            df["\u8bd7\u53e5"] = df["\u8bd7\u53e5"].apply(format_poem)
            st.dataframe(df, use_container_width=True, height=400)
            ra = _random.choice(results)
            st.markdown(f'<div class="ai-reply"><b>AI对曰：</b>「{ra["诗句"]}」<br><span style="color:#666;font-size:0.85rem;">—— {ra["作者"]} · {ra["诗名"]}</span></div>', unsafe_allow_html=True)
        else:
            st.info("无匹配结果")

elif mode == "诗词生成":
    st.markdown("## 诗词生成 · 马尔可夫链")
    style = st.sidebar.selectbox("风格", ["豪放","婉约","田园"])
    if st.sidebar.button("生成", key="gen_btn"):
        with st.spinner("生成中..."):
            mc, ex = train_markov(style)
            poem = markov_gen(mc)
            st.markdown(f'<div class="poem-card" style="text-align:center;font-size:1.2rem;line-height:2;white-space:pre-wrap;">{poem}</div>', unsafe_allow_html=True)
            st.caption(f"风格：{style}")
            st.markdown("**词汇来源**")
            for e in ex[:5]:
                st.markdown(f"- {e[:35]}...")

elif mode == "诗人对比":
    st.markdown("## 诗人对比")
    poets = sorted(set(p["author"] for p in POEMS))
    a = st.sidebar.selectbox("诗人A", poets, index=poets.index("李白") if "李白" in poets else 0)
    b = st.sidebar.selectbox("诗人B", poets, index=poets.index("杜甫") if "杜甫" in poets else min(1,len(poets)-1))
    if st.sidebar.button("对比", key="cp_btn"):
        if a == b:
            st.warning("请选不同诗人")
        else:
            t1, t2, t3 = st.tabs(["高频词","情感分布","统计"])
            with t1:
                wa = poet_words(POEMS, a, 10)
                wb = poet_words(POEMS, b, 10)
                da = pd.DataFrame(wa, columns=["词","频次"]); da["诗人"]=a
                db = pd.DataFrame(wb, columns=["词","频次"]); db["诗人"]=b
                fig = px.bar(pd.concat([da,db]), x="词", y="频次", color="诗人", barmode="group",
                             color_discrete_sequence=["#346538","#9F2F2D"])
                fig.update_layout(plot_bgcolor="#F7F6F3", paper_bgcolor="#F7F6F3")
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                sa = [sentiment(p["content"])["score"] for p in POEMS if p["author"]==a]
                sb = [sentiment(p["content"])["score"] for p in POEMS if p["author"]==b]
                fig2 = px.histogram(pd.DataFrame({"得分":sa+sb,"诗人":[a]*len(sa)+[b]*len(sb)}),
                    x="得分", color="诗人", nbins=12, barmode="overlay", opacity=0.7,
                    color_discrete_sequence=["#346538","#9F2F2D"])
                fig2.update_layout(plot_bgcolor="#F7F6F3", paper_bgcolor="#F7F6F3")
                st.plotly_chart(fig2, use_container_width=True)
            with t3:
                pa = [p for p in POEMS if p["author"]==a]
                pb = [p for p in POEMS if p["author"]==b]
                ta = "".join(p["content"] for p in pa)
                tb = "".join(p["content"] for p in pb)
                stats = [{"指标":"诗作数",a:len(pa),b:len(pb)},
                         {"指标":"总字数",a:len(ta),b:len(tb)},
                         {"指标":"平均句长",a:round(len(ta)/len(pa),1) if pa else 0,b:round(len(tb)/len(pb),1) if pb else 0}]
                st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

elif mode == "情感分析":
    st.markdown("## 情感分析")
    options = [f"{p['title']} - {p['author']}" for p in POEMS]
    if st.sidebar.radio("方式", ["选诗","手动"], key="sa_mode") == "选诗":
        sel = st.sidebar.selectbox("诗作", options)
        idx = options.index(sel)
        text = POEMS[idx]["content"]
    else:
        text = st.sidebar.text_area("输入诗句", "床前明月光 疑是地上霜")
    if st.sidebar.button("分析", key="sa_btn") and text:
        r = sentiment(text)
        disp = (r["score"] + 1) / 2
        col1, col2, col3 = st.columns(3)
        col1.metric("情感指数", f"{disp:.2f}"); col1.progress(disp)
        col2.metric("正面词", r["pos"]); col3.metric("负面词", r["neg"])
        if r["pos_words"]: st.info("正面词: " + " ".join(r["pos_words"]))
        if r["neg_words"]: st.warning("负面词: " + " ".join(r["neg_words"]))
        if r["score"] > 0.2: st.success("倾向: 积极")
        elif r["score"] < -0.2: st.error("倾向: 低沉")
        else: st.info("倾向: 中性")

else:  # Home
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0 1.5rem 0;">
        <div style="font-family: 'Noto Serif SC', serif; font-size: 3rem; font-weight: 700; color: #3A2A1A; letter-spacing: 0.1em;">
            poetry_land
        </div>
        <div style="font-family: 'Noto Serif SC', serif; font-size: 0.9rem; color: #8B7355; letter-spacing: 0.15em; margin-top: 0.3rem;">
            &#10022;  &#x63A2;&#x5343;&#x53E4;&#x8BD7;&#x97F5;  &#x2022;  &#x4F1A;&#x767E;&#x4EE3;&#x98CE;&#x534E;  &#10022;
        </div>
        <div style="margin: 1rem auto; width: 140px; height: 2px; background: linear-gradient(90deg, transparent, #C04040, transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    cards = [
        ("&#x1F338; 飞花令", "搜索含某字的诗句，AI为你对曰作答"),
        ("&#x270D;&#xFE0F; 诗词生成", "马尔可夫链基于古籍训练即兴创作新诗"),
        ("&#x1F465; 诗人对比", "双诗人风格/高频词/情感分布对比"),
        ("&#x1F4AD; 情感分析", "基于情感词典评估诗句情感指数"),
    ]
    for i in range(0, 4, 2):
        c1, c2 = st.columns(2)
        with c1:
            icon, desc = cards[i]
            st.markdown(f"""<div class='poem-card' style='min-height:120px;'><div style='font-size:1.8rem;'>{icon}</div><div style='color:#8B7355; font-size:0.9rem; margin-top:0.4rem;'>{desc}</div></div>""", unsafe_allow_html=True)
        with c2:
            icon, desc = cards[i+1]
            st.markdown(f"""<div class='poem-card' style='min-height:120px;'><div style='font-size:1.8rem;'>{icon}</div><div style='color:#8B7355; font-size:0.9rem; margin-top:0.4rem;'>{desc}</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin:0.5rem 0;'><span style='color:#D4C5A9;'>&#x2500;&#x2500;&#x2500;  &#10022;  &#x2500;&#x2500;&#x2500;</span></div>", unsafe_allow_html=True)

    with st.expander("&#x1F4E5; 加载全量数据"):
        st.markdown("点击按钮将从 GitHub 仓库下载唐诗宋词数据并合并入当前数据集。")
        if st.button("开始下载"):
            with st.spinner("下载中..."):
                new = download_data()
                if new:
                    POEMS.extend(new)
                    total, authors, dynasty_counts, style_counts, top_chars = get_stats(POEMS)
                    st.success(f"已合并 {len(new)} 首")
                else:
                    st.warning("下载失败，请检查网络连接")
# ── Data Dashboard ──
st.markdown("---")
st.markdown("## 数据看板")
c1, c2, c3 = st.columns(3)
c1.metric("总诗数", total); c2.metric("作者数", authors); c3.metric("风格数", style_counts.shape[0])
col_a, col_b = st.columns(2)
with col_a:
    fig = px.pie(dynasty_counts, names="朝代", values="数量", title="朝代分布",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(plot_bgcolor="#F7F6F3", paper_bgcolor="#F7F6F3", margin=dict(t=30,b=0,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)
with col_b:
    st.markdown("**最常用字词 TOP10**")
    st.dataframe(top_chars, use_container_width=True, hide_index=True)

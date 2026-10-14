const fs=require('fs'),cp=require('child_process'),path=require('path'),crypto=require('crypto');
const template='/Users/yrk/.codex/skills/read-paper-xmind-tree/assets/论文解析树模板.xmind';
const output=path.resolve('TAR-ViTPose中心的HPE论文网.xmind');
const dir=fs.mkdtempSync('/tmp/hpe-net-'); cp.execFileSync('unzip',['-q',template,'-d',dir]);
const proto=JSON.parse(fs.readFileSync(path.join(dir,'content.json'),'utf8'))[0];
const id=()=>crypto.randomUUID();
const topic=(title,children=[],opts={})=>{const n={id:id(),class:'topic',title,titleUnedited:false,...opts};if(children.length)n.children={attached:children};return n};
const relation=(a,b,title,type='compare')=>{
 const palettes={
  inherit:{color:'#F28C28',pattern:'solid',arrow:'org.xmind.arrowShape.triangle'},
  compare:{color:'#8B95A1',pattern:'dash',arrow:'org.xmind.arrowShape.triangle'},
  combine:{color:'#24A69A',pattern:'dot',arrow:'org.xmind.arrowShape.triangle'}
 };
 const p=palettes[type];
 const r={id:id(),class:'relationship',end1Id:a.id,end2Id:b.id,controlPoints:{},style:{id:id(),properties:{'line-color':p.color,'line-width':'2pt','line-pattern':p.pattern,'arrow-begin-class':'org.xmind.arrowShape.none','arrow-end-class':p.arrow}}};
 if(title){r.title=title;r.titleUnedited=false} return r;
};
const styled=(title,x,y,color)=>topic(title,[],{position:{x,y},style:{id:id(),properties:{'svg:fill':color,'fill-pattern':'solid','fo:font-weight':'700','fo:font-size':'17pt','shape-class':'org.xmind.topicShape.roundedRect','border-line-width':'0pt'}}});
const sheet=(title,root,relationships=[])=>{const s=structuredClone(proto);s.id=id();s.revisionId=id();s.title=title;s.rootTopic=root;s.relationships=relationships;s.arrangeableLayerOrder=[root.id];s.zones=[];return s};

// Sheet 1: only the network. Details deliberately live on other sheets.
const tar=topic('TAR-ViTPose',[],{style:{id:id(),properties:{'svg:fill':'#FF6268','fill-pattern':'solid','fo:font-weight':'800','fo:font-size':'24pt','shape-class':'org.xmind.topicShape.roundedRect','border-line-width':'0pt'}}});
tar.structureClass='org.xmind.ui.map.unbalanced';
const vit=styled('ViTPose\n单帧空间基线',-650,-330,'#FFB36B');
const std=styled('STDPose\n动态感知 · 稀疏标注',620,-330,'#97D3B6');
const dif=styled('DiffusionPose\n生成热图 · 结构约束',620,330,'#88E2D7');
const pave=styled('PAVE-Net\n端到端 · 多人关联',-650,330,'#6FD0F9');
const concept=(title,x,y,color,shape)=>topic(title,[],{position:{x,y},style:{id:id(),properties:{'svg:fill':color,'fill-pattern':'solid','fo:font-weight':'600','fo:font-size':'13pt','fo:color':'#303840','shape-class':shape,'border-line-color':'#8A7AA8','border-line-width':'1.2pt'}}});
// 菱形：论文与论文之间的关系。
const diamond='org.xmind.topicShape.diamond', ellipse='org.xmind.topicShape.ellipse';
const inheritNode=concept('直接继承\nViT空间骨干',-330,-125,'#EADCF8',diamond);
const alignNode=concept('同一问题的不同机制\nJTA / DAM',300,-155,'#EADCF8',diamond);
const heatmapNode=concept('不同热图范式\n判别式 / 扩散式',300,165,'#EADCF8',diamond);
const paradigmNode=concept('系统范式扩展\nTop-down / 端到端多人',-330,165,'#EADCF8',diamond);
const videoNode=concept('研究演进\n静态表征 → 视频动态',0,-455,'#EADCF8',diamond);
// 椭圆：论文解决问题所采用的具体机制。
const jtaNode=concept('JTA / GRA\n关节时序聚合与恢复',510,0,'#DDF3F0',ellipse);
const damNode=concept('Dynamic-Aware Mask\n选择长距离运动上下文',760,-185,'#DDF3F0',ellipse);
const mambaNode=concept('Mamba–Transformer\n全局长程 / 局部时空关系',760,185,'#DDF3F0',ellipse);
// 圆形：论文共享的研究问题。
const longTemporal=topic('长距离\n时序依赖',[],{position:{x:900,y:0},style:{id:id(),properties:{'svg:fill':'#FFF2CC','fill-pattern':'solid','fo:font-weight':'700','fo:font-size':'13pt','shape-class':'org.xmind.topicShape.circle','border-line-color':'#D6B656','border-line-width':'1.5pt'}}});
const legend=topic('圆角矩形＝论文　圆形＝共同问题　连线文字＝论文关系或解决机制',[],{position:{x:0,y:430},style:{id:id(),properties:{'svg:fill':'#F4F5F7','fill-pattern':'solid','fo:font-size':'12pt','fo:color':'#56606B','shape-class':'org.xmind.topicShape.roundedRect','border-line-width':'0pt'}}});
tar.children={detached:[vit,std,dif,pave,longTemporal,legend]};
const netRelations=[
 relation(vit,tar,'继承 ViT 空间骨干','inherit'),
 relation(tar,std,'JTA 与 DAM：同一问题的不同机制','compare'),
 relation(tar,dif,'判别式融合 与 扩散式生成','compare'),
 relation(tar,pave,'Top-down 到 端到端多人','inherit'),
 relation(vit,std,'静态表征 到 视频动态','inherit'),
 relation(tar,longTemporal,'JTA/GRA：关节时序聚合与恢复','combine'),
 relation(std,longTemporal,'DAM：选择长距离运动上下文','combine'),
 relation(dif,longTemporal,'Mamba-Transformer：全局长程 / 局部时空','combine')
];
const networkSheet=sheet('01 五论文关系网',tar,netRelations);

// Sheet 2: compact cards, no cross-lines.
const card=(name,items)=>topic(name,items.map(x=>topic(x)));
const cardsRoot=topic('五篇论文卡片（细节层）',[
 card('ViTPose · NeurIPS 2022',['任务：单图 top-down 2D HPE','核心：Plain ViT + lightweight decoder','价值：简单、可扩展、训练灵活、知识可迁移','结果：ViTPose-H COCO test-dev 80.9 AP','局限：不显式建模帧间一致性']),
 card('TAR-ViTPose · CVPR 2026',['任务：视频 top-down 2D HPE','JTA：joint query 跨帧聚合关节区域','GRA：时序信息恢复到中心帧全局 token','官方：PoseTrack2017 相比 ViTPose +2.3 mAP','我的复现：ViT-B、5帧、最高 AP 84.209','局限：检测框依赖；结构先验和人物关联仍可加强']),
 card('STDPose · AAAI 2025',['任务：视频 HPE + pose propagation','核心：Dynamic-Aware Mask','核心：视觉特征、热图与 motion dynamics 聚合','价值：降低视频标注需求','结果：PoseTrack2017 报告 87.4 mAP','边界：监督目标和协议与 TAR 不完全相同']),
 card('DiffusionPose · AAAI 2026',['任务：视频 2D HPE；去噪生成热图','核心：Mamba + Transformer','核心：MRF + Markov objective','价值：遮挡关节与骨架结构合理性','代价：多步去噪增加推理成本']),
 card('PAVE-Net · AAAI 2026',['任务：端到端多人视频 2D HPE','核心：Spatial Encoder + STPD + STJD','核心：pose-aware attention 维持同一人物关联','价值：省去检测、crop、NMS','边界：不能与 top-down TAR 直接按 AP 排名'])
]);
cardsRoot.structureClass='org.xmind.ui.map.clockwise';
const cardsSheet=sheet('02 五篇论文卡片',cardsRoot);

// Sheet 3: research synthesis and executable experiments.
const researchRoot=topic('从论文网长出我的研究问题',[
 topic('R1 时序对齐',[topic('TAR：joint query'),topic('STDPose：dynamic-aware mask'),topic('PAVE-Net：person-aware association'),topic('实验 E1：DAM 增强或替代 JTA 固定阈值 mask')]),
 topic('R2 遮挡与结构合理性',[topic('TAR：依赖邻帧视觉证据'),topic('DiffusionPose：MRF / Markov skeleton prior'),topic('实验 E2：为 TAR 加轻量 skeleton consistency，不做完整扩散采样')]),
 topic('R3 数据效率',[topic('STDPose：pose propagation 与伪标签'),topic('ViTPose：预训练与知识迁移'),topic('实验 E3：Sparse-label TAR')]),
 topic('R4 多人端到端',[topic('PAVE-Net：person query + joint decoder'),topic('TAR：top-down joint query'),topic('实验 E4：Person-aware TAR')]),
 topic('公平实验底座',[topic('固定 ViTPose-B、PoseTrack2017、检测框、输入分辨率'),topic('分组：遮挡、快速运动、模糊、拥挤、腕踝'),topic('指标：AP、各关节AP、FPS、GFLOPs、参数量、显存')]),
 topic('优先级',[topic('A：Dynamic-aware JTA'),topic('B：Skeleton-consistent TAR'),topic('C：Sparse-label TAR'),topic('长期：End-to-end multi-person TAR')])
]);
researchRoot.structureClass='org.xmind.ui.map.clockwise';
const researchSheet=sheet('03 研究问题与实验',researchRoot);

// 用户只需要关系网：交付文件中仅保留这一张画布。
fs.writeFileSync(path.join(dir,'content.json'),JSON.stringify([networkSheet]));
const mp=path.join(dir,'metadata.json'),meta=JSON.parse(fs.readFileSync(mp,'utf8'));meta.activeSheetId=networkSheet.id;fs.writeFileSync(mp,JSON.stringify(meta));
if(fs.existsSync(output))fs.unlinkSync(output);
cp.execFileSync('zip',['-q','-r',output,'content.json','metadata.json','manifest.json','Thumbnails'],{cwd:dir});
console.log(output);

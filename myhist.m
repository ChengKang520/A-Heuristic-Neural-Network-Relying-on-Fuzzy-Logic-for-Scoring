function myhist(x) 

[n,y] = hist(x);
maxN = max(n);
y_cor = 1:length(y);

axis([0 1.2 0 maxN+2]);
bar(y_cor,n);

for i = 1:length(y)
    text(y_cor(i),n(i)+40,num2str(n(i)),'VerticalAlignment','middle','HorizontalAlignment','center','fontsize',12,'color',[1 0 0]);
end

set(gca,'xticklabel',y);

% set the label
xtb = get(gca,'XTickLabel');
xt = get(gca,'XTick');
yt = get(gca,'YTick');      

xtextp=xt;                     
ytextp=yt(1)*ones(1,length(xt));

text(xtextp,ytextp,xtb,'HorizontalAlignment','right','rotation',45,'fontsize',10); 

set(gca, 'Position', [0.13,0.390476190476191,0.795,0.534523809523811], 'OuterPosition', [-0.003354838709677,0.318331872626351,1.025806451612903,0.655857434998541])
set(gca,'xticklabel','');% 将原有的标签隐去




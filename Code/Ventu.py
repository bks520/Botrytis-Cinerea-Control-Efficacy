from matplotlib_venn import venn3
import matplotlib.pyplot as plt

# 7个子集的元素数量
subset_sizes = {
    'A': 100,
    'B': 150,
    'C': 200,
    'AB': 30,
    'AC': 40,
    'BC': 60,
    'ABC': 10
}

# 创建一个包含7个子集的韦恩图
venn = venn3(subsets=(subset_sizes['A'], subset_sizes['B'], subset_sizes['AB'],
                      subset_sizes['C'], subset_sizes['AC'], subset_sizes['BC'],
                      subset_sizes['ABC']),
             set_labels=('A', 'B', 'C'))

# 自定义标签和样式
venn.get_label_by_id('100').set_text('A\n(count)')
venn.get_label_by_id('010').set_text('B\n(count)')
venn.get_label_by_id('001').set_text('C\n(count)')
venn.get_label_by_id('110').set_text('AB\n(count)')
venn.get_label_by_id('101').set_text('AC\n(count)')
venn.get_label_by_id('011').set_text('BC\n(count)')
venn.get_label_by_id('111').set_text('ABC\n(count)')

plt.title("7组数据的韦恩图")

# 保存为 PDF 矢量图
plt.savefig("venn_diagram.pdf", format="pdf")

plt.show()

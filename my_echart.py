
import numpy as np
# from flask import Flask, render_template
import pyecharts.options as opts
from pyecharts.charts import Line


class charts:
    def __init__(self):
        self.plot_x = []
        self.no_mask_y = []
        self.mask_y = []
        # self.chart_flage = False

    # 保留两位小数，实现：将列表转数组，再利用numpy 实现保留两位小数，最后将数组转回列表
    def trans_np_2f(self, list_ori):
        mid_np = np.array(list_ori)
        mid_np_2f = np.round(mid_np, 2)
        list_new = list(mid_np_2f)
        return list_new

    def make_chart(self, no_mask_y, mask_y):
        nomask_rate_list = self.trans_np_2f(no_mask_y)
        mask_rate_list = self.trans_np_2f(mask_y)
        print(nomask_rate_list)
        print(mask_rate_list)
        print(len(nomask_rate_list))
        if len(nomask_rate_list) > len(mask_rate_list):
            plot_x = []
            for li in range(len(no_mask_y)):
                plot_x.append(li)
            # self.mychart.get_data(plot_x, nomask_rate_list, mask_rate_list)
        else:
            plot_x = []
            for li in range(len(mask_y)):
                plot_x.append(li)
            # self.mychart.get_data(plot_x, nomask_rate_list, mask_rate_list)
        print("plot_x", plot_x)
        c = (
            Line(init_opts=opts.InitOpts(width="301px", height="201px"))
                .add_xaxis(xaxis_data=plot_x)
                .add_yaxis("no_mask", y_axis=nomask_rate_list, is_smooth=True)
                .add_yaxis("mask", y_axis=mask_rate_list, is_smooth=True)
                .set_series_opts(
                areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title=""),
                xaxis_opts=opts.AxisOpts(
                    axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                    is_scale=False,
                    boundary_gap=False,
                ),
            )
                .render("line_nomask_mask.html")
        )
        return c


if __name__ == "__main__":
    test = charts()
    y1 = []
    y2 = [0.86, 0.83, 0.72, 0.65, 0.56, 0.53, 0.39, 0.27, 0.86, 0.83, 0.86]
    c = test.make_chart(y1, y2)
    print(c)
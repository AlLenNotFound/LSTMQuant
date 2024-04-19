import matplotlib.pyplot as plt
import pandas as pd
import os
import tushare as ts

plt.rcParams['font.sans-serif'] = ['SimHei']
TOKEN1 = 'your token'
ts.set_token(TOKEN1)
pro = ts.pro_api()


def get_data(code, start, end):
    df = pro.daily(ts_code=code, autype='qfq', start_date=start, end_date=end)
    df.index = pd.to_datetime(df.trade_date)
    temp = pro.daily_basic(ts_code=code, autype='qfq', start_date=start, end_date=end, fields='ts_code,trade_date,'
                                                                                              'turnover_rate,ps,'
                                                                                              'volume_ratio,pe,pb,'
                                                                                              'total_mv,dv_ratio')
    temp.index = pd.to_datetime(temp.trade_date)
    factor = pro.stk_factor(ts_code=code, autype='qfq', start_date=start, end_date=end,
                            fields='trade_date,adj_factor,open_hfq,open_qfq,close_hfq,close_qfq,high_hfq,high_qfq,'
                                   'low_hfq,low_qfq,pre_close_hfq,pre_close_qfq,macd_dif,macd_dea,macd,kdj_k,kdj_d,'
                                   'kdj_j,rsi_6,rsi_12,rsi_24,boll_upper,boll_mid,boll_lower,cci')
    # 设置把日期作为索引
    # 定义两个新的列ma和openinterest
    df = df[['open', 'high', 'low', 'close', 'vol']]
    factor_to_add = ['trade_date', 'adj_factor', 'open_hfq', 'open_qfq', 'close_hfq', 'close_qfq', 'high_hfq',
                     'high_qfq', 'low_hfq', 'low_qfq', 'pre_close_hfq', 'pre_close_qfq', 'macd_dif', 'macd_dea',
                     'macd', 'kdj_k', 'kdj_d', 'kdj_j', 'rsi_6', 'rsi_12', 'rsi_24', 'boll_upper', 'boll_mid',
                     'boll_lower', 'cci']
    factor.index = pd.to_datetime(factor.trade_date)
    temp_to_add = ['turnover_rate', 'volume_ratio', 'pe', 'pb', 'ps', 'total_mv', 'dv_ratio']
    print(factor)
    for col in temp_to_add:
        df[col] = temp[col]
    for col in factor_to_add:
        df[col] = factor[col]
    return df


def acquire_code():
    inp_code = input("请输入股票代码:\n")
    inp_start = input("请输入开始时间:'\n'")
    inp_end = input("请输入结束时间:'\n'")
    df = get_data(inp_code, inp_start, inp_end)
    # 输出统计各列的数据量
    print("—" * 30)
    # 分割线
    # print(df.describe())
    # 输出常用统计参数
    df.sort_index(inplace=True)
    path = os.path.join(inp_code + ".csv")
    df.to_csv(path)
    print(df)


# 32模型效果还可以,加了dropout然后11个feature; 128不太好 但数据量很大
acquire_code()

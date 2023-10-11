import streamlit as st
import pyperclip

# 左侧菜单栏
st.sidebar.title("库表设置")

# 多选下拉菜单
selected_options = st.sidebar.multiselect("查询库表", ["派车单明细", "货主画像统计表", "车辆画像表","客户企业站点基础信息表",
                                                      "高德行政区域表","车辆基本属性纬度表","车辆合作关系主表",
                                                      "根节点订单对应运单信息","站点画像指标表","车辆画像指标表"], key="multi_select")

# 确认按钮（左侧）
if st.sidebar.button("确认查询库表"):
    get_table_mapping()


# 右侧内容区域
st.title("sql生成")

# 文本输入框
user_input = st.text_input("你要查询什么？", "")

# 单选框
show_message = st.checkbox("精准查询优化")

# 确认按钮（右侧）
if st.button("生成sql"):
    if show_message:
        extact_prompt ,extact_tables= get_exact_prompt(user_input)
        if extact_prompt is not None:
           st.write("你是要查询如下内容么？")
           if st.button("复制到剪贴板"):
              pyperclip.copy("这是一段提示文本")
    result_sql = gen_sql(user_input)
    st.area(result_sql)

def gen_sql(user_input:str):
    ##sqlDataBaseChain生成sql
    ##返回sql query

def get_exact_prompt(input:str):
    ##语义检索获取相似prompt
    similiary_prompts = aos_knn_search(input)
    ##返回最相似prompt string
    if len(similiary_prompts)>1:
       return similiary_prompts["table_name"],similiary_prompts["query_string"]
    else:
       return None

# 字典映射
value_mapping = {
    "派车单明细": "ads_bi_quality_monitor_shipping_detail",
    "货主画像统计表": "ads_customer_portrait_index_sum_da",
    "车辆画像表": "ads_truck_portrait_index_sum_da",
    "客户企业站点基础信息表":"dim_customer_enterprise_station_base_info",
    "高德行政区域表":"dim_gaode_city_info_v2",
    "车辆基本属性纬度表":"dim_pub_truck_info",
    "车辆合作关系主表":"dim_pub_truck_tenant",
    "根节点订单对应运单信息":"dws_ots_waybill_info_da",
    "站点画像指标表":"dws_station_portrait_index_sum_da",
    "车辆画像指标表":"dws_truck_portrait_index_sum_da"
}

# 确认按钮（映射到字典）
def get_table_mapping():
    selected_values = [value_mapping[option] for option in selected_options]
    st.success("确认查询库表：", selected_values)


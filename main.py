import pandas as pd
r'''
    采用bfs+局部dp的思想
    参数说明:
    dp[当前位置][当前状态]:记录局部最优
    pre[当前位置][当前状态]：用于路径回溯，记录上一个点位置
'''
Inf=2**20
data=pd.read_csv('./tsp.csv')
# print(data)
data.drop(columns='CITY',inplace=True)
num_cities=len(data.loc[0])     #not include 'city' and itself
maxn=2**(num_cities-1)          #所有的状态

dp=[[Inf for _ in range(maxn)] for _ in range(num_cities)]
pre=[[Inf for _ in range(maxn)] for _ in range(num_cities)]
que=[{'state_now':0,'city_now':0}]
dp[0][0]=0
while que:
    state=que.pop(0)
    state_now=state['state_now']
    city_now=state['city_now']
    for city_next in range(1,num_cities):
        if (((1<<(city_next-1))&state_now)==0):
            state_next=((1<<(city_next-1))|state_now)
            if dp[city_next][state_next] == Inf:
                que.append({'state_now':state_next,'city_now':city_next})
            if dp[city_next][state_next]>dp[city_now][state_now]+data.iloc[city_now,city_next]:
                dp[city_next][state_next]=dp[city_now][state_now]+data.iloc[city_now,city_next]
                pre[city_next][state_next]=city_now

ans=Inf
begin=0

r'''
获取最优解
'''
for city in range(1,num_cities):
    if ans>dp[city][maxn-1]+data.iloc[city][0]:
        ans=dp[city][maxn-1]+data.iloc[city][0]
        begin=city
r'''
反向寻找最优路径，由于是对称的，所以正向和逆向路径都可
'''
cur=begin
state_now=maxn-1
print('最优解的一条路径:0->',end='')
while cur!=0:
    print('{}->'.format(cur),end='')
    next_cur=pre[cur][state_now]
    state_now=state_now-(state_now&(1<<(cur-1)))
    cur=next_cur
print('0')
print('最优解为{}'.format(ans))


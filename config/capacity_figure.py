import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 1. Load the Data
csv_data = """capacity_multiplier,algorithm,mean_total_time,backlog_total,renege_rate_obs,throughput
1.0,CEL_BROAD,58.73681893795726,925809,0.5658835546475995,29.15
1.0,FIFO,61.10609143706663,926095,0.6942028985507246,27.6
1.0,LIFO,52.79833632549301,926085,0.7304063860667634,27.56
1.0,PRIORITY,67.39189802394417,926182,0.6455542021924482,16.42
1.5,CEL_BROAD,58.52754908875904,923417,0.5660314276161819,44.56
1.5,FIFO,61.74710138475593,926528,0.6740130782271736,41.29
1.5,LIFO,52.451892525407565,925830,0.7338846995864753,41.11
1.5,PRIORITY,65.12543998027297,925502,0.7237087741132545,32.14
2.0,CEL_BROAD,58.003108304449995,922165,0.5727983850618219,59.26
2.0,FIFO,60.98413077783106,925852,0.6787977548433822,55.23
2.0,LIFO,52.24523100251625,925295,0.731917231276323,54.61
2.0,PRIORITY,63.97999628893722,926477,0.7428800641797032,49.86
2.5,CEL_BROAD,57.59889175222392,920635,0.5754121717891308,73.29
2.5,FIFO,60.63531642227162,926584,0.6731585883731147,68.29
2.5,LIFO,51.80791750004611,926138,0.7310395314787701,68.3
2.5,PRIORITY,63.01015415411502,925143,0.7621858916136846,66.06
3.0,CEL_BROAD,57.42892478334818,920005,0.5789920244357712,87.9
3.0,FIFO,60.88777226544433,925421,0.6790703275529865,83.04
3.0,LIFO,53.11141673947869,925105,0.7329690346083789,82.35
3.0,PRIORITY,62.566651869827304,925815,0.7461500855536544,81.82
4.0,CEL_BROAD,57.32559488568749,916936,0.5724693251533742,116.8
4.0,FIFO,60.36941354393302,924415,0.6818015378982057,109.24
4.0,LIFO,52.58360025652102,923826,0.7343905754417762,110.35
4.0,PRIORITY,61.43001107423859,923039,0.7262350470276687,109.51
6.0,CEL_BROAD,57.38647157045379,910305,0.5649779735682819,176.15
6.0,FIFO,60.34158055298669,922774,0.6848279615407873,165.37
6.0,LIFO,52.3246086798389,922458,0.7391857506361323,165.06
6.0,PRIORITY,60.030227839949134,922386,0.7073141486810551,166.8
8.0,CEL_BROAD,57.12619771505528,904035,0.5636259832529815,235.38
8.0,FIFO,59.944935806048484,921319,0.6830169739003468,219.16
8.0,LIFO,52.10207856145523,921204,0.7359436390732963,221.43
8.0,PRIORITY,59.20064581840417,921401,0.7008727730789994,222.28
10.0,CEL_BROAD,56.64351529096778,898250,0.5580304806565064,293.58
10.0,FIFO,59.87641228566473,921600,0.6834250941756013,276.08
10.0,LIFO,51.76278097469331,921643,0.7348366179738443,274.51
10.0,PRIORITY,58.77227562923376,920625,0.6880002897605854,276.09"""

df = pd.read_csv(io.StringIO(csv_data))
df['renege_rate_obs'] = df['renege_rate_obs'] * 100  # Convert to percentage

# 2. Setup Academic Plot Styling
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("The Futility of Incremental Capacity Increases (1.0x to 10.0x)\nPost-COVID Surge Outpaces Maximum Feasible Hiring", 
             fontsize=16, fontweight='bold', y=0.98)

colors = {
    "FIFO": "#4C72B0", 
    "LIFO": "#DD8452", 
    "PRIORITY": "#55A868", 
    "CEL_BROAD": "#C44E52"
}
markers = {"FIFO": "o", "LIFO": "s", "PRIORITY": "^", "CEL_BROAD": "D"}

# 3. Panel 1: Throughput (The Effort)
sns.lineplot(data=df, x="capacity_multiplier", y="throughput", hue="algorithm", 
             style="algorithm", markers=markers, dashes=False, palette=colors, ax=axes[0, 0], linewidth=2)
axes[0, 0].set_title("Decided Throughput (Cases/Month)\nCourt output scales perfectly linearly", fontsize=12)
axes[0, 0].set_ylabel("Decisions per Month")
axes[0, 0].set_xlabel("Capacity Multiplier (Judges & Officers)")
axes[0, 0].set_xticks([1, 2, 4, 6, 8, 10])

# 4. Panel 2: Backlog (The Crisis) - Forcing Y to start at 0
sns.lineplot(data=df, x="capacity_multiplier", y="backlog_total", hue="algorithm", 
             style="algorithm", markers=markers, dashes=False, palette=colors, ax=axes[0, 1], linewidth=2, legend=False)
axes[0, 1].set_title("True Unresolved Backlog\nThe backlog flatlines despite a 1000% capacity increase", fontsize=12)
axes[0, 1].set_ylabel("Total Cases in Backlog")
axes[0, 1].set_xlabel("Capacity Multiplier")
axes[0, 1].set_ylim(0, 1000000) # Crucial: Starts at 0 to show the flatline
axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
axes[0, 1].set_xticks([1, 2, 4, 6, 8, 10])

# 5. Panel 3: Mean Wait Time (The Human Cost) - Forcing Y to start at 0
sns.lineplot(data=df, x="capacity_multiplier", y="mean_total_time", hue="algorithm", 
             style="algorithm", markers=markers, dashes=False, palette=colors, ax=axes[1, 0], linewidth=2, legend=False)
axes[1, 0].set_title("Mean Wait Time (Months)\nWait times remain rigidly anchored to the structural floor", fontsize=12)
axes[1, 0].set_ylabel("Mean Wait (Months)")
axes[1, 0].set_xlabel("Capacity Multiplier")
axes[1, 0].set_ylim(0, 80) # Crucial: Starts at 0 to show it never drops to 0
axes[1, 0].set_xticks([1, 2, 4, 6, 8, 10])

# 6. Panel 4: Renege Rate (The Attrition)
sns.lineplot(data=df, x="capacity_multiplier", y="renege_rate_obs", hue="algorithm", 
             style="algorithm", markers=markers, dashes=False, palette=colors, ax=axes[1, 1], linewidth=2, legend=False)
axes[1, 1].set_title("Renege / Dropout Rate (%)\nAlgorithms shift attrition, but capacity does not", fontsize=12)
axes[1, 1].set_ylabel("Renege Rate (%)")
axes[1, 1].set_xlabel("Capacity Multiplier")
axes[1, 1].set_ylim(0, 100)
axes[1, 1].set_xticks([1, 2, 4, 6, 8, 10])

# 7. Formatting and Legend Adjustments
handles, labels = axes[0, 0].get_legend_handles_labels()
axes[0, 0].legend_.remove()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=True, title="Queue Scheduling Algorithm")

plt.subplots_adjust(bottom=0.12, hspace=0.3, wspace=0.25)
plt.savefig("fig_capacity_futility.png", dpi=300, bbox_inches='tight')
print("Graph successfully saved as 'fig_capacity_futility.png'")
plt.show()
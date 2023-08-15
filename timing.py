log_str = """
time: load_metadata 0.100554 to run
time: data_lookup 22.71661 to run
time: zero_crossing_lookup 0:08:09.398213 to run
sub_dirs ['combined_subjects_and_trials', 'combined_subjects_and_trials_and_legs', 'graph_each_subject_each_sensor', 'peaks_per_subject', 'pickles', 'poincare', 'similarity_per_subject', 'stats_of_gate_lengths']
gate crossing used -0.3
time: filtering bad gates 0.312415 to run
re-calculate the zero crossings with filter applied
took 0:08:57.989707 to run
time: save_each_subject_each_sensor 88.60467 to run
time: graph_poincare_per_leg  23.650754 to run
time: graph_poincare_comb_leg_per_subject 304.136008 to run
time: poincare_sim_stats_per_sensor 304.290041 to run
time: gate_peak_valley_swing 69.132936 to run
time: calc_shapiro_t_test  69.188376 to run
time: calc_shapiro_t_test_legs_combined 35.33009 to run
time: graph_sensors_combined_subjects_trials 65.158276 to run
time: combined_subjects_trials_signal_stats 55.037448 to run
time: graph_aggregate_subjects_trials_legs 63.339778 to run
time: signal_similarity 66.712302 to run
time: signal_similarity_per_subject_indoor_outdoor 53.580874 to run
time: signal_similarity_per_subject_combined_invsout 53.10164 to run
time: signal_similarity_per_subject_left_right 60.680839 to run
time: lr_control_ivo 0.220665 to run
time: signal_sim_comb_legs 0.018724 to run
saving pickles
normal finished 0:42:32.314290"""
#for line in log_str.splitlines():
#    if 'time:' in line:
#        print(line)
import os, glob
list_scripts = glob.glob("*.py")
for fname in list_scripts:
    ct = 0
    with open(fname, 'r') as fileobj:
        for line in fileobj:
            ct+=1
    print(fname, ct)
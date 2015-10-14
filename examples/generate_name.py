if trials_type  == "P_ARDEGO_Trial":
    run_name = "trials_type_" + trials_type + "_corr_" + corr + "_nsims_" + str(n_sims) + "_parall_" + str(parall)
elif trials_type  == "PSOTrial":
    run_name = "trials_type_" + trials_type + "_corr_" + corr + "_stddv_" + str(max_stdv) + "_sampleon" + sample_on
else:
    run_name = "trials_type_" + trials_type
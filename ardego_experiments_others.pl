sub modify {
    my $tag = $_[0];
    my $newtag = $_[1];
    my $file = $_[2];
    my $command = 'sed -i \'/'.$tag.'/c\\'.$newtag.'\' '.$file;
    system($command);
}

sub time_since_epoch { return `date +%s` }

$n_sims = 50000;
$limit_lambda_search = "False";
$weights_on = "True";

$fitnes_function = "xinyu_rtm";
$run_dir = "examples/${fitnes_function}/"; 
$config_file = $run_dir."configuration_script.py";
$fitness_file = $run_dir."fitness_script.py";
$sampling_fancy = "True";

$results_folder = "/data/mk306/ardego6_${fitnes_function}_software_latin_lambda_${limit_lambda_search}_${n_sims}_${weights_on}_${sampling_fancy}";

#system("rm -Rf $results_folder");
$parallelism = 4;
system("mkdir ".$results_folder); 
$pid = 4;
for ($i=0; $i < $parallelism - 1; $i++) {
    if ($pid) {
        $pid = fork();
    }
    else { 
        last;
    }
}
if ($pid) {
    $i = $parallelism;
}
sleep($i * 5);
$epoch = time_since_epoch();
chomp($epoch);
$tempfile = "/data/mk306/temp_rtm_ardego_$epoch.txt";
$tempfolder = "/data/mk306/ardego_".$epoch."/";
system("mkdir ".$tempfolder); 
system("cp -Rf * ".$tempfolder);
chdir $tempfolder;

print $tempfolder."\n";
system("pwd");
##modify the configuration file to ardego with trails_count
modify("results_folder_path =", 'results_folder_path ="'.$results_folder.'"', $config_file);
modify("trials_type =", 'trials_type =\"P_ARDEGO_Trial\"', $config_file);
modify("surrogate_type =", 'surrogate_type ="bayes2"', $config_file);
modify("trials_count =", "trials_count = 20", $config_file);
modify("n_sims =", "n_sims = ".$n_sims, $config_file);
modify('corr =', 'corr = "anisotropic"', $config_file);
modify('goal =', 'goal = "max"', $config_file);
modify('limit_lambda_search =', "limit_lambda_search = ${limit_lambda_search}", $config_file);
modify('weights_on =', "weights_on = ${weights_on}", $config_file);
modify('sampling_fancy =', "sampling_fancy = ${sampling_fancy}", $config_file);

$j = 0;

#@n_sims = ('n_sims = 1000', 'n_sims = 10000', 'n_sims = 100000'); 
#foreach (@n_sims) {
#    modify($max_error_tag, $_, $fitness_file);
    $p_tag  = "parall =";
    @p = ('parall = 1', 'parall = 2', 'parall = 4', 'parall = 6'); 
    foreach (@p) {
        if ($j % $parallelism == ($i - 1)){ #easiest way to divide tasks
            modify($p_tag, $_, $config_file);
            print "start $j $_ $goal\n";
            my $command = 'make run_example6 > '.$tempfile.' 2>&1';
            system($command);
            print "done $j\n";
        }
        $j = $j + 1;
    }
#}

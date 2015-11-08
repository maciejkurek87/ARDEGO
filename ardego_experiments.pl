sub modify {
    my $tag = $_[0];
    my $newtag = $_[1];
    my $file = $_[2];
    my $command = 'sed -i \'/'.$tag.'/c\\'.$newtag.'\' '.$file;
    system($command);
}

sub time_since_epoch { return `date +%s` }

$n_sims = 5000;
$limit_lambda_search = "False";
$weights_on = "True";
$sampling_fancy = "True";

$run_dir = "examples/quadrature_method_based_app/"; 
$config_file = $run_dir."configuration_script.py";
$fitness_file = $run_dir."fitness_script.py";
$results_folder = "/home/qri/data/ardego_quad_software_latin_lambdaLimit_${limit_lambda_search}_${n_sims}_${weights_on}_${sampling_fancy}";
#system("rm -Rf $results_folder");
$parallelism = 3;
system("mkdir ".$results_folder); 
$pid = 1;
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
$tempfile = "/home/qri/data/temp$epoch.txt";
$tempfolder = "/home/qri/data/ardego_".$epoch."/";
system("mkdir ".$tempfolder); 
system("cp -Rf * ".$tempfolder);
chdir $tempfolder;

print $tempfolder."\n";

##modify the configuration file to ardego with trails_count
modify("results_folder_path =", 'results_folder_path ="'.$results_folder.'"', $config_file);
modify("trials_type =", 'trials_type =\"P_ARDEGO_Trial\"', $config_file);
modify("surrogate_type =", 'surrogate_type ="bayes2"', $config_file);
modify("trials_count =", "trials_count = 1", $config_file);
modify("n_sims =", "n_sims = ${n_sims}", $config_file);
modify('corr =', 'corr = "anisotropic"', $config_file);
modify('limit_lambda_search =', "limit_lambda_search = ${limit_lambda_search}", $config_file);
modify('weights_on =', "weights_on = ${weights_on}", $config_file);
modify('sampling_fancy =', "sampling_fancy = ${sampling_fancy}", $config_file);

$goal_tag  = "goal ="; 
$ret_exec_tag  = "return_execution_time ="; 
@myErrors = ('goal = "max"', "return_execution_time = True"); 
$j = 0;

foreach (@tuple1 = splice(@myErrors,0,2); @tuple1; @tuple1 = splice(@myErrors,0,2)) {
    my ($goal, $ret) = @tuple1;
    ## modify to match goal    
    $max_error_tag  = 'maxError ='; 
    @maxError = ('maxError = 0.1', 'maxError = 0.01', 'maxError = 0.001'); 
    #@maxError = ('maxError = 0.001'); 
    foreach (@maxError) {
        $error = $_;
        #print $error;
        $p_tag  = "parall =";
        #@p = ('parall = 1', 'parall = 2', 'parall = 4', 'parall = 6'); 
        @p = ('parall = 2'); 
        foreach (@p) {
            if ($j % $parallelism == ($i - 1)){ #easiest way to divide tasks
                modify($goal_tag, $goal, $config_file);
                modify($ret_exec_tag, $ret, $fitness_file);
                print $error;
                modify($max_error_tag, $error, $fitness_file);
                modify($p_tag, $_, $config_file);
                print "start $j $_ $goal\n";
                my $command = 'make run_example2 > '.$tempfile.' 2>&1';
                system($command);
                print "done $j\n";
            }
            $j = $j + 1;
        }
    }
}

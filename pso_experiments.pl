sub modify {
    my $tag = $_[0];
    my $newtag = $_[1];
    my $file = $_[2];
    system("sed -i '/$tag/c\\$newtag' ".$file."\n");
}

sub time_since_epoch { return `date +%s` }

$run_dir = "examples/quadrature_method_based_app/"; 
$config_file = $run_dir."configuration_script.py";

$fitness_file = $run_dir."fitness_script.py";

$sampling = "no";
$m = 10;
$results_folder = "/home/qri/data/extra19_pso_quad_params_${sampling}_sample_m_${m}";
system("rm -Rf $results_folder");
$parallelism = 1;
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
$tempfile = "/home/qri/data/temp_pso_quad_${epoch}_${sampling}_${m}.txt";
$tempfolder = '/home/qri/data/pso_'.$epoch.'/';
system('mkdir '.$tempfolder); 
system("cp -Rf * $tempfolder");
chdir $tempfolder;
system("echo `pwd`");
##modify the configuration file to pso with trails_count
modify('results_folder_path =', 'results_folder_path ="'.$results_folder.'"', $config_file);
modify("trials_type =", 'trials_type =\"PSOTrial\"', $config_file);
modify("surrogate_type =", 'surrogate_type ="proper"', $config_file);
modify("classifier =", 'classifier = "DummyClassifier"', $config_file);
modify("trials_count =", "trials_count = 20", $config_file);
modify("sample_on =", 'sample_on = "no"', $config_file);
modify("population_size =", "population_size = 30", $config_file);
modify("M =", "M = ${m}", $config_file);

$goal_tag  = 'goal ='; 
$ret_exec_tag  = 'return_execution_time ='; 
#@myErrors = ('goal = "max"', "return_execution_time = True", 'goal = "min"', "return_execution_time = False"); 
@myErrors = ('goal = "min"', "return_execution_time = False");
$j = 0;
foreach (@tuple1 = splice(@myErrors,0,2); @tuple1; @tuple1 = splice(@myErrors,0,2)) {
    my ($goal, $ret) = @tuple1;
    ## modify to match goal    
    modify($goal_tag, $goal, $config_file);
    modify($ret_exec_tag, $ret, $fitness_file);
    $kernel_tag  = 'corr ='; 
    @kernel = ('corr = "anisotropic"');#, 'corr = "isotropic"', 'corr = "matern3"'); 
    foreach (@kernel) {
        modify($kernel_tag, $_, $config_file);
        $max_error_tag  = 'maxError ='; 
        #@maxError = ('maxError = 0.001','maxError = 0.01','maxError = 0.1');#, 'maxError = 0.001'); 
        @maxError = ('maxError = 0.1'); 
        foreach (@maxError) {
            modify($max_error_tag, $_, $fitness_file);
            $maxErrorSet = $_;
            $max_stdv_tag  = 'max_stdv =';
            $min_stdv_tag  = 'min_stdv =';
            #@stdv = ('max_stdv = 0.5', 'min_stdv = 0.5', 'max_stdv = 0.1', 'min_stdv = 0.1', 'max_stdv = 0.05', 'min_stdv = 0.05', 'max_stdv = 0.01', 'min_stdv = 0.01', 'max_stdv = 0.005', 'min_stdv = 0.005'); 
            @stdv = ('max_stdv = 0.01', 'min_stdv = 0.01');#, 'max_stdv = 0.1', 'min_stdv = 0.1'); 
            foreach (@tuple = splice(@stdv,0,2); @tuple; @tuple = splice(@stdv,0,2)) {
                my ($max_stdv, $min_stdv) = @tuple;
                modify($max_stdv_tag, $max_stdv, $config_file);
                modify($min_stdv_tag, $min_stdv, $config_file);
                if (($j % $parallelism) == ($i - 1)){ #easiest way to divide tasks
                    #print $i."\n";
                    #print "make run_example2 &> $tempfile\n";
                    print "start $j $maxErrorSet $max_stdv \n";
                    system("make run_example2 &> $tempfile");
                    print "done $j\n";
                }
                $j = $j + 1;
            }
        }
    }
}

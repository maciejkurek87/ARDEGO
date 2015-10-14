
sub modify {
    my $tag = $_[0];
    my $newtag = $_[1];
    my $file = $_[2];
    my $command = 'sed -i \'/'.$tag.'/c\\'.$newtag.'\' '.$file;
    system($command);
}

sub time_since_epoch { return `date +%s` }

$run_dir = "examples/quadrature_method_based_app/"; 
$config_file = $run_dir."configuration_script.py";

$fixMax4Bug = "False";
$platformMax3 = "True";

$fitness_file = $run_dir."fitness_script.py";

$results_folder = "/data/mk306/hc_bigFixed_quad_${fixMax4Bug}_${platformMax3}";
#system("rm -Rf $results_folder");
$parallelism = 15;

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
$tempfile = "/data/mk306/hc$epoch.txt";
$tempfolder = '/data/mk306/hc_'.$epoch.'/';
system('mkdir '.$tempfolder); 
system("cp -Rf * $tempfolder");
chdir $tempfolder;
system("echo `pwd`");
##modify the configuration file to pso with trails_count
modify('results_folder_path =', 'results_folder_path ="'.$results_folder.'"', $config_file);
modify("trials_type =", 'trials_type =\"Gradient_Trial\"', $config_file);
modify("surrogate_type =", 'surrogate_type ="bayes2"', $config_file);
modify("trials_count =", "trials_count = 20", $config_file);
modify('fixMax4Bug =', "fixMax4Bug = ${fixMax4Bug}", $fitness_file);
modify('platformMax3 =', "platformMax3 = ${platformMax3}", $fitness_file);

$p_tag  = "parall =";
@p = ('parall = 1', 'parall = 2', 'parall = 4', 'parall = 8','parall = 16'); 
$j = 0;

foreach (@p) {
	modify($p_tag, $_, $config_file);
	$goal_tag  = 'goal ='; 
	$ret_exec_tag  = 'return_execution_time ='; 
	#@myErrors = ('goal = "max"', "return_execution_time = True", 'goal = "min"', "return_execution_time = False"); 
	@myErrors = ('goal = "max"', "return_execution_time = True");
	foreach (@tuple1 = splice(@myErrors,0,2); @tuple1; @tuple1 = splice(@myErrors,0,2)) {
	    my ($goal, $ret) = @tuple1;
	    ## modify to match goal    
	    modify($goal_tag, $goal, $config_file);
	    modify($ret_exec_tag, $ret, $fitness_file);
	    $max_error_tag  = 'maxError ='; 
	    @maxError = ('maxError = 0.1', 'maxError = 0.01', 'maxError = 0.001'); 
	    foreach (@maxError) {
	        modify($max_error_tag, $_, $fitness_file);
	        $maxErrorSet = $_;
	        if (($j % $parallelism) == ($i - 1)){ #easiest way to divide tasks
	            #print $i."\n";
	            #print "make run_example2 &> $tempfile\n";
	            print "start $j $maxErrorSet $max_stdv \n";
		    my $command = 'make run_example2 > '.$tempfile.' 2>&1';
	            system($command);
	            print "done $j\n";
	        }
	        $j = $j + 1;
	    }
	}
}

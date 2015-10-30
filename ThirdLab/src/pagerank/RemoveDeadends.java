package pagerank;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Counters;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import pagerank.GraphToMatrix.Map;
import pagerank.GraphToMatrix.Reduce;

public class RemoveDeadends {

	enum myCounters{ 
		NUMNODES;
	}


	static class Map extends Mapper<LongWritable, Text, Text, Text> {
		
		protected void map(LongWritable key, Text value, Context context) throws  InterruptedException {
			String line = value.toString();
			String[] elem = line.split("\t");
			try {
				context.write(new Text(elem[0]), new Text("S " + elem[1]));
				context.write(new Text(elem[1]), new Text("P " + elem[0]));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	

	static class Reduce extends Reducer<Text, Text, Text, Text> {
		
		protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			Counter c = context.getCounter(myCounters.NUMNODES);
			List<String> valArray = new ArrayList<String>();
			for (Text val : values){
				valArray.add(val.toString());
			}
			List<String> pred = new ArrayList<String>();
			List<String> succ = new ArrayList<String>();
			for(String v : valArray){
				String[] t = v.split("\\s");
				if("P".equals(t[0])){
					pred.add(t[1]);
				}else if("S".equals(t[0])){
					succ.add(t[1]);
				}
			}
			if(!succ.isEmpty()){
				if(!(pred.size()==1 && succ.size()==1 && succ.get(0).equals(key.toString()))){
					c.increment(1);
					for(String p:pred){
						context.write(new Text(p), key);
					}
				}
			}
		}
}

	public static void job(Configuration conf) throws IOException, ClassNotFoundException, InterruptedException{
		
		
		boolean existDeadends = true;
		
		/* You don't need to use or create other folders besides the two listed below.
		 * In the beginning, the initial graph is copied in the processedGraph. After this, the working directories are processedGraphPath and intermediaryResultPath.
		 * The final output should be in processedGraphPath. 
		 */
		FileUtils.copyDirectory(new File(conf.get("graphPath")), new File(conf.get("processedGraphPath")));
		String intermediaryDir = conf.get("intermediaryResultPath");
		String currentInput = conf.get("processedGraphPath");

		FileUtils.deleteQuietly(new File(intermediaryDir));		
		
		long nNodes = conf.getLong("numNodes", 0);
		
		Job job = Job.getInstance(conf);
		job.setJobName("deadends job");
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(Text.class);
		
		job.setMapperClass(Map.class);
		job.setReducerClass(Reduce.class);

		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.setInputPaths(job, new Path(currentInput));
		FileOutputFormat.setOutputPath(job, new Path(intermediaryDir));
		job.waitForCompletion(true);
		
		Counters counters = job.getCounters();
		Counter count = counters.findCounter(myCounters.NUMNODES);
		
		if(count.getValue() == nNodes){
			existDeadends = false;
		}else{
			nNodes = count.getValue();
			conf.setLong("numNodes", nNodes);
		}
		
			
		while(existDeadends)
		{
			/*
			 * Reset Files
			 * 1. Delete the previous input file
			 * 2. Copy the result from the previous iteration to the current input file
			 * 3. Delete the intermediary input file, so as to have a new clean one
			 * */
			FileUtils.deleteDirectory(new File(currentInput));
			FileUtils.copyDirectory(new File(intermediaryDir), new File(currentInput));
			FileUtils.deleteDirectory(new File(intermediaryDir));
			
			job = Job.getInstance(conf);
			job.setJobName("deadends job");
			
			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(Text.class);
			
			job.setMapperClass(Map.class);
			job.setReducerClass(Reduce.class);

			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);

			FileUtils.deleteQuietly(new File(intermediaryDir));
			FileInputFormat.setInputPaths(job, new Path(currentInput));
			FileOutputFormat.setOutputPath(job, new Path(intermediaryDir));
			conf.setLong("numNodes", nNodes);
			job.waitForCompletion(true);
			/* TO DO : configure job and move in the best manner the output for each iteration
			 * you have to update the number of nodes in the graph after each iteration,
			 * use conf.setLong("numNodes", nNodes);
			*/
			
			counters = job.getCounters();
			count = counters.findCounter(myCounters.NUMNODES);
			if (count.getValue() != nNodes){
				nNodes = count.getValue();
				conf.setLong("numNodes", nNodes);
			}else{
				existDeadends = false;	
			}
			
		}				
		
	}
	
}
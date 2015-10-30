package pagerank;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Counter;

import pagerank.RemoveDeadends.myCounters;

/*
 * VERY IMPORTANT 
 * 
 * Each time you need to read/write a file, retrieve the directory path with conf.get 
 * The paths will change during the release tests, so be very carefully, never write the actual path "data/..." 
 * CORRECT:
 * String initialVector = conf.get("initialRankVectorPath");
 * BufferedWriter output = new BufferedWriter(new FileWriter(initialVector + "/vector.txt"));
 * 
 * WRONG
 * BufferedWriter output = new BufferedWriter(new FileWriter(data/initialVector/vector.txt"));
 */

public class PageRank {
	
	public static void createInitialRankVector(String directoryPath, long n) throws IOException 
	{
		File dir = new File(directoryPath);
		FileUtils.deleteQuietly(dir);
		if(!dir.exists()){
			FileUtils.forceMkdir(dir);
		}
		File file = new File(directoryPath + "/vector.txt");
		FileUtils.deleteQuietly(file);
		if(!file.exists()){
			file.createNewFile();
		}
		BufferedWriter output = new BufferedWriter(new FileWriter(file));
		for(int i=1; i<=n; i++){
			output.write(i + new Double(1.0 / n).toString());
			output.newLine();
		}
		output.close();	
	}
	
	public static boolean checkConvergence(String initialDirPath, String iterationDirPath, double epsilon) throws IOException
	{
		List<Double> rInit = new ArrayList<Double>();
		List<Double> rIter = new ArrayList<Double>();
		
		//Read the initial file with the vector and store data into buffRead
		InputStream initialStream = new FileInputStream(initialDirPath + "/vector.txt");
		BufferedReader buffRead = new BufferedReader(new InputStreamReader(initialStream));
		//Read buffRead line by line and add the second element of each line to r0 (rInit)
		String line;
		while((line=buffRead.readLine()) != null){
			String[] elem = line.toString().split("\\s+");
			rInit.add(Double.parseDouble(elem[1]));
		}
		buffRead.close();
		
		//Read the iteration file with the vector and store data into buffRead2
		InputStream iterStream = new FileInputStream(iterationDirPath + "/vector.txt");
		BufferedReader buffRead2 = new BufferedReader(new InputStreamReader(iterStream));
		//Read buffRead2 line by line and add the second element of each line to rk+1 (rIter)
		String line2;
		while((line2=buffRead2.readLine()) != null){
			String[] elem2 = line2.toString().split("\\s+");
			rIter.add(Double.parseDouble(elem2[1]));
		}
		buffRead2.close();
		
		Double sumL1 = 0.0; 
		
		if(rInit.size()==rIter.size()){
			int count = rInit.size();
			for(int i=0; i<count; i++){
				sumL1 += Math.abs(rInit.get(i)-rIter.get(i));
			}
		}
		
		if(sumL1 < epsilon){
			return true;
		}else{
			return false;
		}		
	}
	
	public static void avoidSpiderTraps(String vectorDirPath, long nNodes, double beta)
	{
		HashMap<Integer,Double> myGraph = new HashMap<Integer,Double>();
		
		try{
			//Read the initial file with the vector and store data into buffRead
			InputStream vectorStream= new FileInputStream(vectorDirPath + "/vector.txt");
			BufferedReader buffRead = new BufferedReader(new InputStreamReader(vectorStream));
			//Read buffRead line by line and add the second element of each line to r0 (rInit)
			String line;
			while((line=buffRead.readLine()) != null){
				String[] elem = line.toString().split("\\s+");
				myGraph.put(Integer.parseInt(elem[0]),Double.parseDouble(elem[1]));
			}
			buffRead.close();
		}catch(Exception e){}
		
		
		try{
			//Reset the file where we are going to write the data
			File file = new File(vectorDirPath + "/vector.txt");
			FileUtils.forceDelete(file);
			if(!file.exists()){
				file.createNewFile();
			}
			//Read the file and reset the value o each line to include the Random Teleports with beta
			BufferedWriter buffWriter = new BufferedWriter(new FileWriter(vectorDirPath + "/vector.txt"));
			for(int i = 0; i<myGraph.size(); i++){
				Double val = new Double((beta*(myGraph.get(i+1)) + (double)(1.0-beta)/nNodes));
				buffWriter.write((i+1) + " " + val.toString());
				buffWriter.newLine();
			}
			buffWriter.close();
		}catch(IOException e){
			e.printStackTrace();
		}
		
	}
	
	public static void iterativePageRank(Configuration conf) 
			throws IOException, InterruptedException, ClassNotFoundException
	{
		
		
		String initialVector = conf.get("initialVectorPath");
		String currentVector = conf.get("currentVectorPath");
		
		String finalVector = conf.get("finalVectorPath"); 
		/*here the testing system will search for the final rank vector*/
		
		Double epsilon = conf.getDouble("epsilon", 0.1);
		Double beta = conf.getDouble("beta", 0.8);

		RemoveDeadends.job(conf);
		
		Long nNodes = conf.getLong("numNodes", 1);
		createInitialRankVector(initialVector, nNodes);
		GraphToMatrix.job(conf);
		
		boolean conv = false;
		MatrixVectorMult.job(conf);
		
		avoidSpiderTraps(currentVector, nNodes, beta);
		
		while(!conv){
			FileUtils.deleteQuietly(new File(initialVector));
			FileUtils.moveFile(new File(currentVector + "/vector.txt"), new File(initialVector + "/vector.txt"));
			
			MatrixVectorMult.job(conf);
			
			avoidSpiderTraps(currentVector, nNodes, beta);
			
			conv = checkConvergence(initialVector, currentVector, epsilon);
			
		}
		
		FileUtils.copyDirectory(new File(currentVector), new File(finalVector));
		//TO DO

		// to retrieve the number of nodes use long nNodes = conf.getLong("numNodes", 0); 

		

		// when you finished implementing delete this line
		//throw new UnsupportedOperationException("Implementation missing");
		
	}
}


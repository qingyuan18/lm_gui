import boto3
import json
import requests
import time
from collections import defaultdict
from requests_aws4auth import AWS4Auth
import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain
from langchain.tools.base import BaseTool, Tool, tool
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
    CallbackManagerForChainRun
)
from langchain.llms.bedrock import Bedrock

class CustomerizedSQLDatabaseChain(SQLDatabaseChain):

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        _run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        if table_names_to_use is None:
            table_names_to_use=self.database._include_tables
            print("table_names_to_use==")
            print(table_names_to_use)
        table_info = self.database.get_table_info(table_names=table_names_to_use)

        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "dialect": self.database.dialect,
            "table_info": table_info,
            #"stop": ["\nSQLResult:"],
        }
        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs)  # input: sql generation
            sql_cmd = self.llm_chain.predict(
                callbacks=_run_manager.get_child(),
                **llm_inputs,
            ).strip()
            print("orginal sql_cmd=="+sql_cmd)
            if self.return_sql:
                return {self.output_key: sql_cmd}
            if not self.use_query_checker:
                _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
                intermediate_steps.append(
                    sql_cmd
                )  # output: sql generation (no checker)
                #########定制sqlcoder 模型输出##############
                pattern = r"SQLQuery: (.*?)\n"
                matches = re.findall(pattern, sql_cmd)
                match = matches[1]
                sql_cmd = match
                #print("query sql=="+sql_cmd)

                intermediate_steps.append({"sql_cmd": sql_cmd})  # input: sql exec
                result = self.database.run(sql_cmd)
                intermediate_steps.append(str(result))  # output: sql exec
            else:
                query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query", "dialect"]
                )
                query_checker_chain = LLMChain(
                    llm=self.llm_chain.llm, prompt=query_checker_prompt
                )
                query_checker_inputs = {
                    "query": sql_cmd,
                    "dialect": self.database.dialect,
                }
                checked_sql_command: str = query_checker_chain.predict(
                    callbacks=_run_manager.get_child(), **query_checker_inputs
                ).strip()
                intermediate_steps.append(
                    checked_sql_command
                )  # output: sql generation (checker)
                _run_manager.on_text(
                    checked_sql_command, color="green", verbose=self.verbose
                )
                intermediate_steps.append(
                    {"sql_cmd": checked_sql_command}
                )  # input: sql exec
                result = self.database.run(checked_sql_command)
                intermediate_steps.append(str(result))  # output: sql exec
                sql_cmd = checked_sql_command

            _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            _run_manager.on_text(result, color="yellow", verbose=self.verbose)
            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            if self.return_direct:
                final_result = result
            else:
                _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
                llm_inputs["input"] = input_text
                intermediate_steps.append(llm_inputs)  # input: final answer
                final_result = self.llm_chain.predict(
                    callbacks=_run_manager.get_child(),
                    **llm_inputs,
                ).strip()
                intermediate_steps.append(final_result)  # output: final answer
                _run_manager.on_text(final_result, color="green", verbose=self.verbose)
            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
            return chain_result
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            raise exc

def run_query(query):
    infos=""
    table_name=""
    question=""
    ####post process agent action output######
    if "\n" in query:
        infos = query.split("\n")
    elif "," in query:
        infos = query.split(",")

    if ":" in infos[0]:
        table_name=infos[0].split(":")[1]
    else:
        table_name=infos[0]

    if ":" in infos[1]:
        question=infos[1].split(":")[1]
    else:
        question=infos[1]

    table_name=table_name.strip()
    question=question.strip()

    db_chain = CustomerizedSQLDatabaseChain.from_llm(sm_sql_llm, db, verbose=True, return_sql=True,return_intermediate_steps=False)

    if table_name is not None:
        db_chain.database._include_tables=[table_name]
    response=db_chain.run(question)
    return response

def aos_knn_search(client, field,q_embedding, index, size=1):
    if not isinstance(client, OpenSearch):
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = {
        "size": size,
        "query": {
            "knn": {
                field: {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    opensearch_knn_respose = []
    query_response = client.search(
        body=query,
        index=index
    )
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'query_desc_text':item['_source']['query_desc_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose

def aos_reverse_search(client, index_name, field, query_term, exactly_match=False, size=1):
    """
    search opensearch with query.
    :param host: AOS endpoint
    :param index_name: Target Index Name
    :param field: search field
    :param query_term: query term
    :return: aos response json
    """
    if not isinstance(client, OpenSearch):
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = None
    if exactly_match:
        query =  {
            "query" : {
                "match_phrase":{
                    "doc": {
                        "query": query_term,
                        "analyzer": "ik_smart"
                      }
                }
            }
        }
    else:
        query = {
            "size": size,
            "query": {
                "query_string": {
                "default_field": "query_desc_text",
                "query": query_term
              }
            },
           "sort": [{
               "_score": {
                   "order": "desc"
               }
           }]
    }
    query_response = client.search(
        body=query,
        index=index_name
    )
    result_arr = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'query_desc_text':item['_source']['query_desc_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return result_arr




def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    parameters = {
    }

    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                "parameters": parameters,
                "is_query" : True,
                "instruction" :  "为这个句子生成表示以用于检索相关文章："
            }
        ),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['sentence_embeddings']
    return embeddings




def get_topk_items(opensearch_query_response, topk=5):
    opensearch_knn_nodup = []
    unique_ids = set()
    for item in opensearch_query_response:
        if item['id'] not in unique_ids:
            opensearch_knn_nodup.append(item['score'], item['idx'],item['database_name'],item['table_name'],item['query_desc_text'])
            unique_ids.add(item['id'])
    return opensearch_knn_nodup



def k_nn_ingestion_by_aos(docs,index,hostname,username,passwd):
    auth = (username, passwd)
    search = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        ##http_auth = awsauth ,
        http_auth = auth ,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    for doc in docs:
        query_desc_embedding = doc['query_desc_embedding']
        database_name = doc['database_name']
        table_name = doc['table_name']
        query_desc_text = doc["query_desc_text"]
        document = { "query_desc_embedding": query_desc_embedding, 'database_name':database_name, "table_name": table_name,"query_desc_text":query_desc_text}
        search.index(index=index, body=document)
import streamlit as st
import googleapiclient.discovery
import pandas as pd
from datetime import datetime
import os
from io import BytesIO
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

# import openai
from anthropic import Anthropic
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# 앱 설정
st.set_page_config(
    page_title="Youtube video analyser",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None, 
        'About': None
    }
)

# Configure API keys
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# CLAUDE_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize APIs
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# YouTube Transcript API로 스크립트로 요약
def youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        first_minute = []
        current_time = 0
        
        for line in transcript:
            if current_time > 180:  # 60초 = 1분
                break
            first_minute.append(line['text'])
            current_time += line['duration']
            
        return ' '.join(first_minute)
    except:
        return ''

# # YouTube Data API v3의 captions 엔드포인트로 요약
# def get_captions(video_id):
#     captions_request = youtube.captions().list(
#         part='snippet',
#         videoId=video_id
#     )
#     return captions_request.execute()

# 유튜브 동영상 기본 정보 불러오기
def fetch_youtube_data(search_query, max_results=50):
    """Fetch YouTube video data for given search query"""
    videos_data = []
    
    # Initial search request
    request = youtube.search().list(
        q=search_query,
        part='snippet',
        type='video',
        maxResults=min(50, max_results)
    )
    response = request.execute()
    
    # Process video IDs to get statistics
    video_ids = [item['id']['videoId'] for item in response['items']]
    stats_request = youtube.videos().list(
        part='statistics',
        id=','.join(video_ids)
    )
    stats_response = stats_request.execute()

    # Get channel IDs for subscriber count
    channel_ids = [item['snippet']['channelId'] for item in response['items']]
    channels_request = youtube.channels().list(
        part='statistics',
        id=','.join(set(channel_ids))  # 중복 제거
    )
    channels_response = channels_request.execute()
    
    # 채널 ID를 구독자 수와 매핑
    channel_subscribers = {
        channel['id']: int(channel['statistics'].get('subscriberCount', 0))
        for channel in channels_response['items']
    }
    
    # Combine video data with statistics
    for video, stats in zip(response['items'], stats_response['items']):
        views = int(stats['statistics'].get('viewCount', 0))

        if views >= 1000:
            video_id = video['id']['videoId']
            channel_id = video['snippet']['channelId']
            subscriber_count = channel_subscribers.get(channel_id, 0)
            
            # 조회수/구독자수 비율 계산 (구독자가 0인 경우 처리)
            view_sub_ratio = (views / subscriber_count) * 100 if subscriber_count > 0 else 0

            # 동영상 스크립트
            script = youtube_transcript(video_id)
            
            videos_data.append({
                'title': video['snippet']['title'], 
                'channel': video['snippet']['channelTitle'], 
                'publishedAt': video['snippet']['publishedAt'], 
                'views': views, 
                'subscribers': subscriber_count, 
                'view_sub_ratio': round(view_sub_ratio, 2),  # 소수점 2자리까지 표시
                'likes': int(stats['statistics'].get('likeCount', 0)), 
                'comments': int(stats['statistics'].get('commentCount', 0)), 
                'description': video['snippet']['description'], 
                'url': f"https://www.youtube.com/watch?v={video['id']['videoId']}", 
                'thumbnail': video['snippet']['thumbnails']['high']['url'], 
                '1min_script': script  # 1분 요약 스크립트
            })
    
    return pd.DataFrame(videos_data)

# 분석
def analyze_with_llm(df, query, context=None):
    """Analyze YouTube data using Claude"""
    # Prepare data summary for Claude
    data_summary = df.to_string()
    
    # RAG로 불러온 context가 있는 경우 프롬프트에 추가
    rag_context = f"\n\nPDF 문서 관련 컨텍스트:\n{context}" if context else ""
    
    prompt = f"""당신은 유튜브 데이터 분석 전문가입니다. 데이터를 기반으로 통찰력 있는 분석을 제공합니다.
다음은 YouTube 검색 결과 데이터 분석을 위한 정보입니다:
검색어: {query}
총 영상 수: {len(df)}

데이터:
{data_summary}
{rag_context}

다음 사항들을 고려하여 입력된 키워드를 주제로한 동영상 제목과 스크립트를 추천해주세요:
1. 조회수, 좋아요, 댓글 수의 전반적인 트렌드
2. 가장 인기 있는 영상들의 공통점
3. 주요 채널들과 그들의 컨텐츠 특징
4. 검색어와 관련된 콘텐츠 트렌드
5. 시청자 참여도가 높은 영상의 특징

참고할만한 통계:
- 평균 조회수: {df['views'].mean():,.0f}
- 평균 좋아요: {df['likes'].mean():,.0f}
- 평균 댓글수: {df['comments'].mean():,.0f}
- 최다 조회수: {df['views'].max():,}
- 최다 좋아요: {df['likes'].max():,}"""
    
    try:
        message = client.chat.completions.create(
            model='o1-mini', 
            messages=[
                # {'role': 'system', 'content': "당신은 유튜브 데이터 분석 전문가입니다. 데이터를 기반으로 통찰력 있는 분석을 제공합니다."}, 
                {'role': 'user', 'content': prompt}
            ], 
            max_completion_tokens=3000,  # max_tokens=1000, 
            # temperature=0.7, 
        )

        print('API 응답:', message)
        
        if message:
            return message.choices[0].message.content
        else:
            print("API 응답이 비어있습니다.")
            return None
    
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        
        return None


# RAG
def load_pdf(file_stream):
    # PDF 읽기
    pdf = PdfReader(file_stream)  # 파일을 로컬에서 불러와야 하는 문제로 인해 다른 메소드 사용
    
    # 텍스트 추출
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Document 객체 생성 및 분할
    doc = Document(page_content=text)
    documents = text_splitter.split_documents([doc])
    
    return documents

def rag(documents):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)

    return vectorstore

def relevant_context(vectorstore, query, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])

    return context


def main():
    st.title("🎥 YouTube 트렌드 분석기")
    st.write("키워드를 입력하면 관련 유튜브 영상을 Claude AI로 분석해드립니다.")
    
    # PDF 파일 업로드
    upload = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    
    # Search input
    search_query = st.text_input("분석하고 싶은 키워드를 입력하세요:")
    max_results = st.slider("분석할 영상 수", 10, 50, 30)

    # PDF 파일 입력
    vectorstore = None
    if upload:
        with st.spinner("PDF 파일을 처리중입니다..."):
            # 입력된 파일 그냥
            pdf_stream = BytesIO(upload.getvalue())
            
            chunks = load_pdf(pdf_stream)
            vectorstore = rag(chunks)
            st.success("PDF 처리가 완료되었습니다!")
    
    
    if st.button("분석 시작"):
        with st.spinner("데이터를 수집하고 분석 중입니다..."):
            try:
                # Fetch YouTube data
                df = fetch_youtube_data(search_query, max_results)

                # RAG로 컨텍스트 가져오기
                context = None
                if vectorstore:
                    context = relevant_context(vectorstore, search_query)
                
                # Display basic statistics
                st.subheader("📊 기본 통계")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 조회수", f"{df['views'].sum():,}")
                with col2:
                    st.metric("평균 좋아요", f"{int(df['likes'].mean()):,}")
                with col3:
                    st.metric("평균 댓글", f"{int(df['comments'].mean()):,}")
                
                # Show top videos
                st.subheader("🔥 인기 영상 TOP 5")
                top_videos = df.nlargest(5, 'views')

                # 만 단위로 나타내기
                def format_to_10k(n):
                    num = round(n / 10000, 1)  # 만 단위로 변환하고 소수점 첫째자리에서 반올림

                    return f"{num}만"

                top_videos['views'] = top_videos['views'].apply(lambda x: format_to_10k(x) + " 건")
                top_videos['subscribers'] = top_videos['subscribers'].apply(lambda x: format_to_10k(x) + " 명")
                top_videos['view_sub_ratio'] = top_videos['view_sub_ratio'].apply(lambda x: f"{round(x)}%")  # 정수로 반올림
                
                # 썸네일에 링크 추가
                top_videos['thumbnail'] = top_videos.apply(lambda x: f'<a href="{x["url"]}" target="_blank"><img src="{x["thumbnail"]}" width="240"/></a>', axis=1)
                # top_videos['thumbnail'] = top_videos['thumbnail'].apply(lambda x: x)

                # 보여줄 컬럼 선택 및 이름 변경 (링크 열 제외)
                display_videos = top_videos[['thumbnail', 'title', 'channel', 'views', 'subscribers', 'view_sub_ratio', '1min_script']]
                display_videos.columns = ['썸네일', '제목', '채널명', '조회수', '구독자수', '조회수/구독자 비율', '최초 1분 스크립트']

                # HTML을 허용하는 방식으로 데이터프레임 표시
                st.markdown(display_videos.to_html(escape=False, index=False), unsafe_allow_html=True)
                # st.dataframe(display_videos, hide_index=True)  # 좌우 스크롤 가능. 단, 썸네일 이미지 표시 불가.
                # st.dataframe(
                #     display_videos,
                #     hide_index=True,
                #     column_config={
                #         "썸네일": st.column_config.ImageColumn(
                #             "썸네일",
                #             help="클릭하면 영상으로 이동합니다",
                #             width="medium"
                #         ),
                #         "제목": st.column_config.TextColumn(
                #             "제목",
                #             width="medium"
                #         ),
                #         "최초 1분 스크립트": st.column_config.TextColumn(
                #             "최초 1분 스크립트",
                #             width="large"
                #         )
                #     }
                # )
                
                # Engagement rate calculation
                df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'] * 100
                
                # Additional statistics
                st.subheader("📈 참여도 분석")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평균 참여율", f"{df['engagement_rate'].mean():.2f}%")
                with col2:
                    st.metric("최고 참여율", f"{df['engagement_rate'].max():.2f}%")
                
                # Claude Analysis
                st.subheader("🤖 Claude AI 분석 리포트")
                analysis = analyze_with_llm(df, search_query, context)  # 분석!
                st.markdown(analysis)
                print('analysis:\n', analysis)
                
                # Data visualization
                st.subheader("📊 데이터 시각화")
                tab1, tab2 = st.tabs(["조회수 분포", "참여도 분포"])
                
                with tab1:
                    st.bar_chart(df.nlargest(10, 'views')[['title', 'views']].set_index('title'))
                
                with tab2:
                    st.bar_chart(df.nlargest(10, 'engagement_rate')[['title', 'engagement_rate']].set_index('title'))
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()

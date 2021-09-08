//메뉴 색상 바꾸기
$(window).on('scroll',changeColor);
function changeColor(){
  let offset=$(window).scrollTop();
  if(offset>=661){ //about section 밑으로 내려옴
    $('#about-btn').css('color','#4A4A4A');
    $('#contact-btn').css('color','#4A4A4A');
    //막대기 애니메이션
$('.skill').each(function(){
    let skill=$(this);
    let percentage=skill.find('.percentage').text();
    skill.find('.inner-bar').animate({width: percentage},1200);
  }
);
  }
  else if(offset<661){
    $('#about-btn').css('color','white');
    $('#contact-btn').css('color','white');
  }
}
//매뉴 클릭시 위치 이동
$('#about-btn').on('click',function() {
  $('html,body').animate({scrollTop: $('.about').position().top },1000);
});
$('#contact-btn').on('click',function() {
  $('html,body').animate({scrollTop: $('.contact').position().top },1000);
});
//내용 나타내기(효과o)
function showContent(){
  //스크롤 시 실행되는 함수를 scrollHandler 함수 한개로 합치자
  $('section').each(function(){
    if($(window).scrollTop()>= $(this).position().top){
      $(this).find('.vertical-center').animate({opacity:1,top:0},1000);
    }
    
  });
}
$(window).on('scroll',showContent);
showContent();


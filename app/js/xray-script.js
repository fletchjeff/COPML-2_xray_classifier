    let url = 'https://modelservice.ml-18a296af-d86.demo-aws.ylcu-atmi.cloudera.site/model';
    let api_key_1 = "my87udph3c9up73g7nwv3b3pdt1hzf4h"
    let api_key_2 = "mgp9zye1k3rfx7hik70oeaibk4tw08rk"
    let image_name = "img_test_pneumonia"

    function go_fetch(img_name) {
      d3.select("#loader").attr("style","display:block")
      let image_data = getBase64Image(document.getElementById(img_name))
      let post_data_1 = {
        accessKey: api_key_1,
        request: {
          image: image_data
        }
      };

      fetch(url, {
          method: 'POST', // or 'PUT'
          body: JSON.stringify(post_data_1), // data can be `string` or {object}!
          headers: {
            'Content-Type': 'application/json'
          }
        })
        .then(res => res.json())
        .then(function (response) {
          d3.select("#prediction_model_1").text(response.response.prediction + " (" + response.response.prediction_value.toFixed(5) + ")")
          if (response.response.prediction != "normal" ){
                    let post_data_2 = {
                    accessKey: api_key_2,
                    request: {
                      image: image_data
                    }
                  };

                  fetch(url, {
                      method: 'POST', // or 'PUT'
                      body: JSON.stringify(post_data_2), // data can be `string` or {object}!
                      headers: {
                        'Content-Type': 'application/json'
                      }
                    })
                    .then(res => res.json())
                    .then(function (response) {
                      d3.select("#prediction_model_2").text(response.response.prediction + " (" + response.response.prediction_value.toFixed(5) + ")")
                      d3.select("#loader").attr("style","display:none")
                    })
                    .catch(error => console.error('Error:', error));  
              }
           else {
             d3.select("#loader").attr("style","display:none")
           }
        })
        .catch(error => console.error('Error:', error));
          
    }

    function getBase64Image(img) {
      var canvas = document.createElement("canvas");
      scaling = document.getElementById("img_test_normal").naturalWidth / 1000
      canvas.width = img.naturalWidth / scaling;
      canvas.height = img.naturalHeight / scaling;
      var ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, img.naturalWidth, img.naturalHeight, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/png");
    }

    function get_new_image() {
      d3.select("#loader").attr("style","display:block")
      d3.select("#prediction_model_1").text("...")
      d3.select("#prediction_model_2").text("...")
      fetch(window.location.origin + "/random_image", {
          method: 'GET', // or 'PUT'
          headers: {
            'Content-Type': 'application/json'
          }
        })
        .then(res => res.json())
        .then(function (response) {
          d3.select("#img_test_normal").attr("src", "/" + response.file)
          d3.select("#actual_value").html(response.file.substring(response.file.indexOf("/")+6,response.file.lastIndexOf("/")))
//          d3.select("#prediction").text("Prediction : ")
//          d3.select("#prediction_value").text("Prediction Value : ")
          d3.select("#loader").attr("style","display:none")
        })
        .catch(error => console.error('Error:', error));
    }